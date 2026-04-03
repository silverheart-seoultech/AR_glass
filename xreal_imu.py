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
        self._P = np.eye(7) * 0.1
        self._P[4:, 4:] = np.eye(3) * 0.01

        # Process noise (tuned for ICM-42688-P at 1kHz)
        self._q_gyro = 5e-5       # gyro noise — lower = trust gyro more
        self._q_gyro_bias = 1e-9  # bias random walk — very slow

        # Measurement noise
        self._r_accel = 1.0       # base accel noise (adaptively scaled)

        self._last_timestamp_us = None
        self._sample_idx = 0
        self._warmup_samples = 200  # ~0.2s at 1kHz
        self._calibrated = False

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

    def calibrate_static(self, duration_sec: float = 1.5):
        """
        Static calibration: collect data while stationary to estimate
        gyro bias and initial orientation from gravity.

        Call this after connect(), before stream().
        The glasses should be worn and the head held still.
        """
        if not self._imu_device:
            raise RuntimeError("Not connected. Call connect() first.")

        print(f"[CAL] Hold still for {duration_sec:.1f}s...")

        self._send_imu_enable(True)
        time.sleep(0.05)

        gyro_samples = []
        accel_samples = []
        start = time.monotonic()

        while time.monotonic() - start < duration_sec:
            data = self._imu_device.read(64, timeout_ms=100)
            if not data:
                continue
            imu = self._parse_imu_packet(data)
            if imu is None:
                continue
            gyro_samples.append([imu.gyro_x, imu.gyro_y, imu.gyro_z])
            accel_samples.append([imu.accel_x, imu.accel_y, imu.accel_z])

        self._send_imu_enable(False)
        time.sleep(0.05)
        self._drain_buffer()

        if len(gyro_samples) < 100:
            print("[CAL] Not enough samples, skipping calibration")
            return

        gyro_arr = np.array(gyro_samples)
        accel_arr = np.array(accel_samples)

        # ── 1. Gyro bias: mean of stationary gyro readings ──
        gyro_bias = gyro_arr.mean(axis=0)
        gyro_std = gyro_arr.std(axis=0)

        # ── 2. Initial orientation from gravity direction ──
        accel_mean = accel_arr.mean(axis=0)
        a_norm = np.linalg.norm(accel_mean)

        if a_norm > 0.01:
            # Gravity in body frame → compute Roll & Pitch
            # (Yaw is unobservable from accel alone — set to 0)
            ax, ay, az = accel_mean / a_norm

            # We want: R(q) * [0,0,1] = [ax, ay, az]  (gravity direction)
            pitch = math.asin(-ax)
            roll = math.atan2(ay, az)
            yaw = 0.0  # unobservable

            # Euler → quaternion
            cr, sr = math.cos(roll/2), math.sin(roll/2)
            cp, sp = math.cos(pitch/2), math.sin(pitch/2)
            cy, sy = math.cos(yaw/2), math.sin(yaw/2)

            q0 = cr*cp*cy + sr*sp*sy
            q1 = sr*cp*cy - cr*sp*sy
            q2 = cr*sp*cy + sr*cp*sy
            q3 = cr*cp*sy - sr*sp*cy

            init_q = np.array([q0, q1, q2, q3])
            init_q /= np.linalg.norm(init_q)
        else:
            init_q = np.array([1.0, 0.0, 0.0, 0.0])

        # ── 3. Set EKF initial state ──
        self._x[0:4] = init_q
        self._x[4:7] = gyro_bias

        # Tight initial covariance — we're confident in these values
        self._P = np.eye(7) * 1e-4
        self._P[4:, 4:] = np.eye(3) * (gyro_std ** 2).mean() * 0.1

        # Skip warmup since we're already calibrated
        self._calibrated = True

        print(
            f"[CAL] Done ({len(gyro_samples)} samples)\n"
            f"      Gyro bias: ({gyro_bias[0]:+.5f}, {gyro_bias[1]:+.5f}, {gyro_bias[2]:+.5f}) rad/s\n"
            f"      Gyro std:  ({gyro_std[0]:.5f}, {gyro_std[1]:.5f}, {gyro_std[2]:.5f}) rad/s\n"
            f"      Init Roll: {math.degrees(roll):+.1f}°  Pitch: {math.degrees(pitch):+.1f}°\n"
            f"      |accel|:   {a_norm:.3f} m/s² (expect ~{GRAVITY:.3f})"
        )

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
            if not self._calibrated and self._sample_idx <= self._warmup_samples:
                r_scale = 0.01  # trust accel heavily during warmup (no calibration)
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

        # ════════════════════ UPDATE: ZUPT (gyro bias when stationary) ════
        # When stationary: gyro should read zero → direct observation of bias
        gyro_mag = np.linalg.norm(gyro)
        is_stationary = (accel_dev < 0.3) and (gyro_mag < 0.08)

        zupt_ready = self._calibrated or (self._sample_idx > self._warmup_samples)
        if is_stationary and zupt_ready:
            # Measurement: gyro = bias + noise → z = gyro, h = bg
            z_zupt = gyro
            h_zupt = x[4:7]
            y_z = z_zupt - h_zupt

            # Jacobian: d(gyro)/d(state) = [0(3x4), I(3x3)]
            H_z = np.zeros((3, 7))
            H_z[:, 4:7] = np.eye(3)

            R_z = np.eye(3) * 1e-4  # tight: we're confident it's stationary
            S_z = H_z @ P @ H_z.T + R_z
            K_z = P @ H_z.T @ np.linalg.inv(S_z)

            dx = K_z @ y_z
            x[0:4] += dx[0:4]
            x[0:4] /= np.linalg.norm(x[0:4])
            x[4:7] += dx[4:7]

            IKH = np.eye(7) - K_z @ H_z
            P = IKH @ P @ IKH.T + K_z @ R_z @ K_z.T

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

        # Static calibration: hold still for gyro bias + initial orientation
        # Skip with --no-cal flag
        if "--no-cal" not in sys.argv:
            imu.calibrate_static(duration_sec=1.5)

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
