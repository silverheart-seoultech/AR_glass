"""
XREAL Air (NR-7100RGL) IMU Data Reader

Reads raw accelerometer and gyroscope data from the XREAL Air AR glasses
via USB HID interface, with optional Madgwick sensor fusion for orientation.

References:
  - ar-drivers-rs: https://github.com/badicsalex/ar-drivers-rs
  - Protocol blog: https://voidcomputing.hu/blog/worse-better-prettier/
"""

import struct
import math
import time
import json
import signal
import sys
import os
import glob
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

# USB IDs
XREAL_VENDOR_ID = 0x3318
XREAL_AIR_PRODUCT_ID = 0x0424

# HID Interface numbers
IMU_INTERFACE = 3
MCU_INTERFACE = 4

# IMU protocol constants
IMU_HEADER = bytes([0x01, 0x02])
IMU_ENABLE_CMD = 0x19
IMU_CONFIG_LEN_CMD = 0x14
IMU_CONFIG_DATA_CMD = 0x15
IMU_PACKET_HEADER = 0xAA

GRAVITY = 9.80665
DEG_TO_RAD = math.pi / 180.0


@dataclass
class IMUData:
    """Parsed IMU sensor reading."""
    timestamp_us: int
    gyro_x: float  # rad/s
    gyro_y: float  # rad/s
    gyro_z: float  # rad/s
    accel_x: float  # m/s^2
    accel_y: float  # m/s^2
    accel_z: float  # m/s^2
    mag_x: float = 0.0  # µT (micro-Tesla)
    mag_y: float = 0.0
    mag_z: float = 0.0
    mag_valid: bool = False

    def __str__(self):
        s = (
            f"t={self.timestamp_us:>12d}us | "
            f"Gyro: ({self.gyro_x:+8.4f}, {self.gyro_y:+8.4f}, {self.gyro_z:+8.4f}) rad/s | "
            f"Accel: ({self.accel_x:+8.3f}, {self.accel_y:+8.3f}, {self.accel_z:+8.3f}) m/s²"
        )
        if self.mag_valid:
            s += f" | Mag: ({self.mag_x:+7.1f}, {self.mag_y:+7.1f}, {self.mag_z:+7.1f}) µT"
        return s


@dataclass
class Orientation:
    """Fused orientation from Madgwick filter."""
    quaternion: np.ndarray  # [w, x, y, z]
    euler_deg: np.ndarray   # [roll, pitch, yaw] in degrees

    def __str__(self):
        r, p, y = self.euler_deg
        return f"Roll: {r:+7.2f}°  Pitch: {p:+7.2f}°  Yaw: {y:+7.2f}°"


def _read_i24_le(data: bytes, offset: int) -> int:
    """Read a signed 24-bit little-endian integer."""
    b0 = data[offset]
    b1 = data[offset + 1]
    b2 = data[offset + 2]
    val = b0 | (b1 << 8) | (b2 << 16)
    if val & 0x800000:
        val -= 0x1000000
    return val


def _crc32(data: bytes) -> int:
    """Compute CRC32 (same as zlib/Ethernet, matching ar-drivers-rs crc32_adler)."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF


def _build_imu_command(cmd_id: int, data: bytes = b"") -> bytes:
    """
    Build an IMU command packet (header 0xAA).

    Packet layout (64 bytes total):
      [0]     head      = 0xAA
      [1:5]   checksum  = CRC32-LE over bytes [5 .. 5+length)
      [5:7]   length    = len(data) + 3  (LE)
      [7]     cmd_id
      [8..]   data payload
      rest    zero-padded to 64 bytes
    """
    length = len(data) + 3  # 2 bytes length-field + 1 byte cmd_id ... actually: length covers itself(2) + cmd(1) + data
    packet = bytearray(64)
    packet[0] = IMU_PACKET_HEADER
    # [1:5] checksum - filled after
    struct.pack_into("<H", packet, 5, length)
    packet[7] = cmd_id & 0xFF
    for i, b in enumerate(data):
        packet[8 + i] = b

    # CRC32 over bytes [5 .. 5+length)
    crc_data = bytes(packet[5:5 + length])
    crc = _crc32(crc_data)
    struct.pack_into("<I", packet, 1, crc)

    return bytes(packet)


class HIDRawDevice:
    """Thin wrapper around /dev/hidraw for read/write with timeout."""

    def __init__(self, path: str):
        self.path = path
        self._fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)

    def write(self, data: bytes):
        os.write(self._fd, data)

    def read(self, size: int = 64, timeout_ms: int = 1000) -> Optional[bytes]:
        import select
        timeout_sec = timeout_ms / 1000.0
        r, _, _ = select.select([self._fd], [], [], timeout_sec)
        if r:
            return os.read(self._fd, size)
        return None

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class XREALAirIMU:
    """Interface for XREAL Air IMU data via USB HID."""

    def __init__(self):
        self._imu_device: Optional[HIDRawDevice] = None
        self._mcu_device: Optional[HIDRawDevice] = None
        self._running = False
        self._calibration = None

        # ── EKF State: [q0, q1, q2, q3, bg_x, bg_y, bg_z] ──
        # q = orientation quaternion [w,x,y,z]
        # bg = gyroscope bias (rad/s)
        self._x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._P = np.eye(7) * 0.1  # state covariance
        self._P[4:, 4:] = np.eye(3) * 0.01  # gyro bias initial uncertainty

        # Process noise
        self._q_gyro = 1e-4       # gyro noise (rad/s)²
        self._q_gyro_bias = 1e-8  # gyro bias random walk

        # Measurement noise (base values, adaptively scaled)
        self._r_accel = 0.5       # accel measurement noise
        self._r_mag = 1.0         # mag measurement noise

        # Reference magnetic field (learned from first valid samples)
        self._mag_ref = None
        self._mag_ref_samples = 0
        self._mag_ref_accum = np.zeros(3)

        self._last_timestamp_us = None
        self._sample_idx = 0
        self._warmup_samples = 200  # ~0.2s at 1kHz

    def _find_device_paths(self) -> tuple[Optional[str], Optional[str]]:
        """Find hidraw device paths for IMU and MCU interfaces via sysfs."""
        import re
        imu_path = None
        mcu_path = None
        vid_pid = f"{XREAL_VENDOR_ID:08X}:{XREAL_AIR_PRODUCT_ID:08X}"

        for hidraw in sorted(glob.glob("/dev/hidraw*")):
            name = os.path.basename(hidraw)
            try:
                with open(f"/sys/class/hidraw/{name}/device/uevent") as f:
                    if vid_pid not in f.read().upper():
                        continue
                real = os.path.realpath(f"/sys/class/hidraw/{name}/device")
                m = re.search(r':1\.(\d+)', real)
                if m:
                    iface = int(m.group(1))
                    if iface == IMU_INTERFACE:
                        imu_path = hidraw
                    elif iface == MCU_INTERFACE:
                        mcu_path = hidraw
            except (FileNotFoundError, PermissionError):
                continue

        return imu_path, mcu_path

    def connect(self):
        """Connect to the XREAL Air glasses."""
        imu_path, mcu_path = self._find_device_paths()

        if imu_path is None:
            raise ConnectionError(
                "XREAL Air not found. Check:\n"
                "  1. Glasses are connected via USB-C\n"
                "  2. udev rules are installed (sudo cp 99-xreal-air.rules /etc/udev/rules.d/)\n"
                "  3. Run: sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                "  4. Reconnect the glasses\n"
                "  5. Or run with sudo for testing"
            )

        self._imu_device = HIDRawDevice(imu_path)
        print(f"[OK] IMU device connected: {imu_path}")

        if mcu_path:
            self._mcu_device = HIDRawDevice(mcu_path)
            print(f"[OK] MCU device connected: {mcu_path}")

    def disconnect(self):
        """Disconnect from the glasses."""
        self._running = False
        try:
            self._send_imu_enable(False)
        except Exception:
            pass
        if self._imu_device:
            self._imu_device.close()
            self._imu_device = None
        if self._mcu_device:
            self._mcu_device.close()
            self._mcu_device = None
        print("[OK] Disconnected")

    def _send_imu_enable(self, enable: bool):
        """Enable or disable IMU data streaming."""
        cmd = _build_imu_command(IMU_ENABLE_CMD, bytes([0x01 if enable else 0x00]))
        self._imu_device.write(cmd)

    def _drain_buffer(self):
        """Drain any pending packets from the IMU device."""
        while True:
            data = self._imu_device.read(64, timeout_ms=50)
            if not data:
                break

    def read_calibration(self) -> Optional[dict]:
        """Read IMU calibration data from the device."""
        if not self._imu_device:
            return None

        # Ensure IMU stream is off and drain any buffered data
        self._send_imu_enable(False)
        time.sleep(0.1)
        self._drain_buffer()

        # Request config length (cmd 0x14)
        cmd = _build_imu_command(IMU_CONFIG_LEN_CMD)
        self._imu_device.write(cmd)
        time.sleep(0.2)

        resp = self._imu_device.read(64, timeout_ms=1000)
        if not resp or len(resp) < 12 or resp[0] != IMU_PACKET_HEADER:
            print("[WARN] Could not read calibration length")
            return None

        # Response: [0]=0xAA [1:5]=crc [5:7]=length [7]=cmd [8:12]=config_len(u32 LE)
        config_len = struct.unpack_from("<I", bytes(resp), 8)[0]
        if config_len == 0 or config_len > 65536:
            print(f"[WARN] Unexpected calibration length: {config_len}")
            return None

        print(f"[INFO] Reading calibration data ({config_len} bytes)...")

        # Read config data in chunks (max 48 bytes payload per packet)
        config_data = bytearray()
        read_offset = 0
        while read_offset < config_len:
            chunk_size = min(48, config_len - read_offset)
            cmd = _build_imu_command(
                IMU_CONFIG_DATA_CMD,
                struct.pack("<HH", read_offset, chunk_size),
            )
            self._imu_device.write(cmd)
            time.sleep(0.01)

            resp = self._imu_device.read(64, timeout_ms=1000)
            if not resp or len(resp) < 9 or resp[0] != IMU_PACKET_HEADER:
                break

            # Payload data starts at offset 8
            data_start = 8
            data_end = min(data_start + chunk_size, len(resp))
            config_data.extend(resp[data_start:data_end])
            read_offset += chunk_size

        try:
            config_str = config_data.decode("utf-8", errors="ignore").rstrip("\x00")
            self._calibration = json.loads(config_str)
            print("[OK] Calibration loaded")
            return self._calibration
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Failed to parse calibration: {e}")
            return None

    def _parse_imu_packet(self, data: bytes) -> Optional[IMUData]:
        """Parse a raw 64-byte IMU HID packet into IMUData."""
        if len(data) < 42:
            return None
        if data[0] != 0x01 or data[1] != 0x02:
            return None

        # Timestamp (nanoseconds -> microseconds)
        timestamp_ns = struct.unpack_from("<Q", bytes(data), 4)[0]
        timestamp_us = timestamp_ns // 1000

        # Gyroscope
        gyro_mul = struct.unpack_from("<H", bytes(data), 12)[0]
        gyro_div = struct.unpack_from("<I", bytes(data), 14)[0]
        if gyro_div == 0:
            gyro_div = 1

        gyro_x_raw = _read_i24_le(data, 18)
        gyro_y_raw = _read_i24_le(data, 21)
        gyro_z_raw = _read_i24_le(data, 24)

        gyro_scale = (gyro_mul / gyro_div) * DEG_TO_RAD
        # Axis remapping: negate X, swap Y/Z
        gyro_x = -gyro_x_raw * gyro_scale
        gyro_y = gyro_z_raw * gyro_scale
        gyro_z = gyro_y_raw * gyro_scale

        # Accelerometer
        accel_mul = struct.unpack_from("<H", bytes(data), 27)[0]
        accel_div = struct.unpack_from("<I", bytes(data), 29)[0]
        if accel_div == 0:
            accel_div = 1

        accel_x_raw = _read_i24_le(data, 33)
        accel_y_raw = _read_i24_le(data, 36)
        accel_z_raw = _read_i24_le(data, 39)

        accel_scale = (accel_mul / accel_div) * GRAVITY
        # Axis remapping: negate X, swap Y/Z
        accel_x = -accel_x_raw * accel_scale
        accel_y = accel_z_raw * accel_scale
        accel_z = accel_y_raw * accel_scale

        # Apply calibration bias if available
        if self._calibration:
            gyro_bias = self._calibration.get("gyro_bias", [0, 0, 0])
            accel_bias = self._calibration.get("accel_bias", [0, 0, 0])
            if len(gyro_bias) == 3:
                gyro_x -= gyro_bias[0]
                gyro_y -= gyro_bias[1]
                gyro_z -= gyro_bias[2]
            if len(accel_bias) == 3:
                accel_x -= accel_bias[0]
                accel_y -= accel_bias[1]
                accel_z -= accel_bias[2]

        # Magnetometer (big-endian multiplier/divisor, bizarre i16 encoding)
        mag_x, mag_y, mag_z = 0.0, 0.0, 0.0
        mag_valid = False
        if len(data) >= 54:
            mag_mul = struct.unpack_from(">H", bytes(data), 42)[0]
            mag_div = struct.unpack_from(">I", bytes(data), 44)[0]
            if mag_div != 0 and mag_mul != 0:
                def _bizarre_i16(d, off):
                    """XOR 0x80 on high byte, then interpret as signed i16."""
                    lo = d[off]
                    hi = d[off + 1] ^ 0x80
                    return struct.unpack("<h", bytes([lo, hi]))[0]

                mag_x_raw = _bizarre_i16(data, 48)
                mag_y_raw = _bizarre_i16(data, 50)
                mag_z_raw = _bizarre_i16(data, 52)

                mag_scale = mag_mul / mag_div
                # Axis remapping (same as accel/gyro)
                mag_x = -mag_x_raw * mag_scale
                mag_y = mag_z_raw * mag_scale
                mag_z = mag_y_raw * mag_scale
                mag_valid = True

        return IMUData(
            timestamp_us=timestamp_us,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z, mag_valid=mag_valid,
        )

    @staticmethod
    def _quat_mult(a, b):
        """Hamilton quaternion product [w,x,y,z]."""
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])

    @staticmethod
    def _quat_rot(q, v):
        """Rotate vector v by quaternion q: q ⊗ [0,v] ⊗ q*."""
        qv = np.array([0.0, v[0], v[1], v[2]])
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        tmp = np.array([
            q[0]*qv[0] - q[1]*qv[1] - q[2]*qv[2] - q[3]*qv[3],
            q[0]*qv[1] + q[1]*qv[0] + q[2]*qv[3] - q[3]*qv[2],
            q[0]*qv[2] - q[1]*qv[3] + q[2]*qv[0] + q[3]*qv[1],
            q[0]*qv[3] + q[1]*qv[2] - q[2]*qv[1] + q[3]*qv[0],
        ])
        res = np.array([
            tmp[0]*qc[0] - tmp[1]*qc[1] - tmp[2]*qc[2] - tmp[3]*qc[3],
            tmp[0]*qc[1] + tmp[1]*qc[0] + tmp[2]*qc[3] - tmp[3]*qc[2],
            tmp[0]*qc[2] - tmp[1]*qc[3] + tmp[2]*qc[0] + tmp[3]*qc[1],
            tmp[0]*qc[3] + tmp[1]*qc[2] - tmp[2]*qc[1] + tmp[3]*qc[0],
        ])
        return res[1:4]

    @staticmethod
    def _quat_to_euler(q):
        """Quaternion [w,x,y,z] → Euler [roll, pitch, yaw] in degrees."""
        w, x, y, z = q
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)

        return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])

    def _ekf_update(self, imu: IMUData) -> Orientation:
        """
        7-state EKF AHRS: state = [q(4), gyro_bias(3)]

        Predict:  quaternion integration with bias-corrected gyro
        Update 1: accelerometer → gravity direction (Roll/Pitch)
        Update 2: magnetometer → magnetic north (Yaw) — if available

        Adaptive: measurement noise scaled by accel deviation from gravity
        """
        self._sample_idx += 1

        # ── dt ──
        if self._last_timestamp_us is None:
            self._last_timestamp_us = imu.timestamp_us
            dt = 0.001
        else:
            dt = (imu.timestamp_us - self._last_timestamp_us) / 1_000_000.0
            self._last_timestamp_us = imu.timestamp_us
        if dt <= 0 or dt > 1.0:
            dt = 0.001

        x = self._x.copy()
        P = self._P.copy()
        q = x[0:4]
        bg = x[4:7]

        gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])
        accel = np.array([imu.accel_x, imu.accel_y, imu.accel_z])

        # ════════════════════ PREDICT ════════════════════
        # Bias-corrected angular velocity
        w_corr = gyro - bg

        # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, w_corr]
        wx, wy, wz = w_corr
        Omega = 0.5 * np.array([
            [0,  -wx, -wy, -wz],
            [wx,  0,   wz, -wy],
            [wy, -wz,  0,   wx],
            [wz,  wy, -wx,  0 ],
        ])

        # Integrate quaternion
        q_new = q + Omega @ q * dt
        q_new /= np.linalg.norm(q_new)

        # State transition Jacobian F (7x7)
        F = np.eye(7)
        F[0:4, 0:4] = np.eye(4) + Omega * dt

        # d(q_dot)/d(bg) = -0.5 * [q]_right * I_ext * dt
        # Simplified: bias affects quaternion through gyro
        qw, qx, qy, qz = q
        dq_dbg = -0.5 * dt * np.array([
            [-qx, -qy, -qz],
            [ qw,  qz, -qy],
            [-qz,  qw,  qx],
            [ qy, -qx,  qw],
        ])
        F[0:4, 4:7] = dq_dbg

        # Process noise Q
        Q = np.zeros((7, 7))
        Q[0:4, 0:4] = np.eye(4) * self._q_gyro * dt * dt
        Q[4:7, 4:7] = np.eye(3) * self._q_gyro_bias * dt

        # Predicted state and covariance
        x[0:4] = q_new
        # bg stays the same (random walk model)
        P = F @ P @ F.T + Q

        # ════════════════════ UPDATE: ACCELEROMETER ════════════════════
        a_norm = np.linalg.norm(accel)
        if a_norm > 0.01:
            # Expected gravity in body frame: R(q)^T * [0,0,g]
            # = rotate [0,0,1] by q_conjugate
            q_conj = np.array([q_new[0], -q_new[1], -q_new[2], -q_new[3]])
            g_body = self._quat_rot(q_conj, np.array([0.0, 0.0, GRAVITY]))

            # Measurement: normalized accel
            z_accel = accel / a_norm
            h_accel = g_body / np.linalg.norm(g_body)

            # Innovation
            y_a = z_accel - h_accel

            # Jacobian H_accel (3x7): d(h_accel)/d(state)
            # Approximated numerically for robustness
            H_a = np.zeros((3, 7))
            eps = 1e-6
            for j in range(4):
                x_plus = x.copy()
                x_plus[j] += eps
                x_plus[0:4] /= np.linalg.norm(x_plus[0:4])
                qc_p = np.array([x_plus[0], -x_plus[1], -x_plus[2], -x_plus[3]])
                gp = self._quat_rot(qc_p, np.array([0.0, 0.0, GRAVITY]))
                gp_n = gp / np.linalg.norm(gp)
                H_a[:, j] = (gp_n - h_accel) / eps

            # Adaptive R: scale up when accel deviates from gravity
            accel_dev = abs(a_norm - GRAVITY)
            if self._sample_idx <= self._warmup_samples:
                r_scale = 0.01  # trust accel heavily during warmup
            elif accel_dev > 1.5:
                r_scale = 100.0  # strong linear accel → distrust accel
            elif accel_dev > 0.3:
                r_scale = 1.0 + (accel_dev - 0.3) / 1.2 * 99.0
            else:
                r_scale = 1.0

            R_a = np.eye(3) * self._r_accel * r_scale

            # Kalman gain
            S_a = H_a @ P @ H_a.T + R_a
            K_a = P @ H_a.T @ np.linalg.inv(S_a)

            # State update
            dx = K_a @ y_a
            x[0:4] += dx[0:4]
            x[0:4] /= np.linalg.norm(x[0:4])
            x[4:7] += dx[4:7]

            # Covariance update (Joseph form for numerical stability)
            IKH = np.eye(7) - K_a @ H_a
            P = IKH @ P @ IKH.T + K_a @ R_a @ K_a.T

        # ════════════════════ UPDATE: MAGNETOMETER ════════════════════
        if imu.mag_valid:
            mag = np.array([imu.mag_x, imu.mag_y, imu.mag_z])
            mag_norm = np.linalg.norm(mag)

            if mag_norm > 1.0:  # valid reading (µT)
                # Learn magnetic reference from first stable samples
                if self._mag_ref is None:
                    if abs(a_norm - GRAVITY) < 0.5:  # only when stationary
                        self._mag_ref_accum += mag
                        self._mag_ref_samples += 1
                        if self._mag_ref_samples >= 100:
                            # Project reference to horizontal plane using current q
                            avg_mag = self._mag_ref_accum / self._mag_ref_samples
                            q_cur = x[0:4]
                            mag_world = self._quat_rot(q_cur, avg_mag)
                            # Keep only horizontal component (zero out vertical)
                            self._mag_ref = np.array([
                                math.sqrt(mag_world[0]**2 + mag_world[1]**2),
                                0.0,
                                mag_world[2],
                            ])
                            print(f"[OK] Magnetic reference set: {self._mag_ref}")

                if self._mag_ref is not None:
                    q_cur = x[0:4]
                    q_conj = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])

                    # Expected mag in body frame
                    h_mag = self._quat_rot(q_conj, self._mag_ref)
                    h_mag_n = h_mag / np.linalg.norm(h_mag)

                    z_mag = mag / mag_norm

                    y_m = z_mag - h_mag_n

                    # Numerical Jacobian
                    H_m = np.zeros((3, 7))
                    for j in range(4):
                        x_plus = x.copy()
                        x_plus[j] += eps
                        x_plus[0:4] /= np.linalg.norm(x_plus[0:4])
                        qc_p = np.array([x_plus[0], -x_plus[1], -x_plus[2], -x_plus[3]])
                        mp = self._quat_rot(qc_p, self._mag_ref)
                        mp_n = mp / np.linalg.norm(mp)
                        H_m[:, j] = (mp_n - h_mag_n) / eps

                    R_m = np.eye(3) * self._r_mag
                    S_m = H_m @ P @ H_m.T + R_m
                    K_m = P @ H_m.T @ np.linalg.inv(S_m)

                    dx = K_m @ y_m
                    x[0:4] += dx[0:4]
                    x[0:4] /= np.linalg.norm(x[0:4])
                    x[4:7] += dx[4:7]

                    IKH = np.eye(7) - K_m @ H_m
                    P = IKH @ P @ IKH.T + K_m @ R_m @ K_m.T

        # ── Store state ──
        self._x = x
        self._P = P

        euler = self._quat_to_euler(x[0:4])
        return Orientation(quaternion=x[0:4].copy(), euler_deg=euler)

    def read_one(self, timeout_ms: int = 1000) -> Optional[IMUData]:
        """Read a single IMU sample. Returns None on timeout."""
        if not self._imu_device:
            raise RuntimeError("Not connected. Call connect() first.")

        data = self._imu_device.read(64, timeout_ms=timeout_ms)
        if not data:
            return None
        return self._parse_imu_packet(data)

    def stream(
        self,
        callback: Optional[Callable[[IMUData, Optional[Orientation]], None]] = None,
        enable_fusion: bool = True,
        duration_sec: float = 0,
    ):
        """
        Stream IMU data continuously.

        Args:
            callback: Called for each IMU sample with (imu_data, orientation).
                      If None, prints to stdout.
            enable_fusion: If True, runs Madgwick filter for orientation.
            duration_sec: Stop after this many seconds (0 = run forever).
        """
        if not self._imu_device:
            raise RuntimeError("Not connected. Call connect() first.")

        # Enable IMU streaming
        self._send_imu_enable(True)
        print("[OK] IMU streaming enabled")
        time.sleep(0.05)

        self._running = True
        start_time = time.monotonic()
        sample_count = 0

        try:
            while self._running:
                if duration_sec > 0 and (time.monotonic() - start_time) >= duration_sec:
                    break

                data = self._imu_device.read(64, timeout_ms=100)
                if not data:
                    continue

                imu = self._parse_imu_packet(data)
                if imu is None:
                    continue

                orientation = None
                if enable_fusion:
                    orientation = self._ekf_update(imu)

                sample_count += 1

                if callback:
                    callback(imu, orientation)
                else:
                    if orientation:
                        print(f"{imu}  |  {orientation}")
                    else:
                        print(imu)

        except KeyboardInterrupt:
            pass
        finally:
            self._send_imu_enable(False)
            elapsed = time.monotonic() - start_time
            rate = sample_count / elapsed if elapsed > 0 else 0
            print(f"\n[OK] Stopped. {sample_count} samples in {elapsed:.1f}s ({rate:.0f} Hz)")

    def stop(self):
        """Stop the streaming loop."""
        self._running = False


def main():
    imu = XREALAirIMU()

    def signal_handler(sig, frame):
        print("\nStopping...")
        imu.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect
        imu.connect()

        # Calibration is disabled by default — the 38KB read destabilizes
        # the device if interrupted. Enable with --calibration flag if needed.
        if "--calibration" in sys.argv:
            try:
                imu.read_calibration()
            except Exception as e:
                print(f"[WARN] Calibration skipped: {e}")

        # Stream IMU data with sensor fusion
        print("\n--- Streaming IMU data (Ctrl+C to stop) ---\n")
        imu.stream(enable_fusion=True)

    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected: {e}")
        sys.exit(1)
    finally:
        imu.disconnect()


if __name__ == "__main__":
    main()
