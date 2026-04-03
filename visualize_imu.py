"""
XREAL Air IMU 3D Orientation Visualizer

글래스의 Roll/Pitch/Yaw를 실시간 3D 박스로 시각화하여
IMU 센서 퓨전 값이 실제로 유효한지 검증한다.

Usage:
    python3 -u visualize_imu.py
"""

import threading
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from xreal_imu import XREALAirIMU, IMUData, Orientation


# ── 글래스 형태의 3D 꼭짓점 ──────────────────────────
# Sensor coordinate system (from IMU data after reset):
#   When identity quaternion (all angles=0):
#   X = sensor X (remapped), Y = sensor Y, Z = sensor Z
#
# Box shape: wide glasses facing +Z
#   X = left-right (wide), Y = up-down (medium), Z = front-back (thin)
W, H, D = 1.6, 0.5, 0.3  # width(X), height(Y), depth(Z)

VERTICES = np.array([
    [-W/2, -H/2, -D/2],
    [+W/2, -H/2, -D/2],
    [+W/2, +H/2, -D/2],
    [-W/2, +H/2, -D/2],
    [-W/2, -H/2, +D/2],
    [+W/2, -H/2, +D/2],
    [+W/2, +H/2, +D/2],
    [-W/2, +H/2, +D/2],
])

FACES = [
    [0, 1, 2, 3],  # back
    [4, 5, 6, 7],  # front
    [0, 1, 5, 4],  # bottom
    [2, 3, 7, 6],  # top
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5],  # right
]

FACE_COLORS = [
    (0.6, 0.6, 0.6, 0.5),  # back: gray
    (0.2, 0.6, 1.0, 0.7),  # front: blue (렌즈 방향)
    (0.8, 0.8, 0.8, 0.3),  # bottom
    (0.8, 0.8, 0.8, 0.3),  # top
    (1.0, 0.4, 0.4, 0.5),  # left: red
    (0.4, 1.0, 0.4, 0.5),  # right: green
]


def rotation_matrix_from_quat(q):
    """
    Quaternion [w,x,y,z] → 3x3 rotation matrix for visualization.

    Uses the quaternion directly (no euler decomposition issues).
    Applies a fixed remap so that the box displays as:
      - (0,0,0) orientation = box upright, facing forward
      - Pitch(nod) = box tilts forward/backward
      - Roll(tilt) = box tilts left/right
      - Yaw(turn)  = box rotates left/right
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R


class IMUVisualizer:
    def __init__(self):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.euler_deg = np.zeros(3)  # [roll, pitch, yaw] in degrees
        self.gyro = np.zeros(3)
        self.accel = np.zeros(3)
        self.sample_count = 0
        self.hz = 0.0
        self.lock = threading.Lock()
        self._running = True

    def imu_callback(self, imu_data: IMUData, orientation: Orientation):
        with self.lock:
            self.quat = orientation.quaternion.copy()
            self.euler_deg = orientation.euler_deg.copy()
            self.gyro = np.array([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])
            self.accel = np.array([imu_data.accel_x, imu_data.accel_y, imu_data.accel_z])
            self.sample_count += 1

    def run(self):
        # ── IMU 스레드 시작 ──
        imu = XREALAirIMU()
        try:
            imu.connect()
            imu.calibrate_static(duration_sec=1.5)
        except ConnectionError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)

        def imu_thread():
            try:
                imu.stream(callback=self.imu_callback, enable_fusion=True)
            except Exception as e:
                print(f"[IMU thread error] {e}")

        t = threading.Thread(target=imu_thread, daemon=True)
        t.start()

        # 데이터 수신 대기
        print("[INFO] Waiting for IMU data...")
        for _ in range(50):
            if self.sample_count > 0:
                break
            time.sleep(0.1)

        if self.sample_count == 0:
            print("[ERROR] No IMU data received. Try unplugging and re-plugging the glasses.")
            imu.stop()
            imu.disconnect()
            sys.exit(1)

        print(f"[OK] Receiving data. Launching 3D view...")
        print(f"     Press 'R' in the plot window to reset orientation (episode reset)\n")

        # ── Matplotlib 3D 시각화 ──
        plt.ion()
        fig = plt.figure(figsize=(14, 6))
        fig.canvas.manager.set_window_title("XREAL Air IMU Visualizer")

        # Keyboard handler: 'r' = reset orientation
        def on_key(event):
            if event.key in ('r', 'R'):
                imu.reset_orientation()
                # Clear time series on reset
                t_buf.clear()
                roll_buf.clear()
                pitch_buf.clear()
                yaw_buf.clear()
                nonlocal t_start
                t_start = time.monotonic()

        fig.canvas.mpl_connect('key_press_event', on_key)

        # 왼쪽: 3D 박스
        ax3d = fig.add_subplot(121, projection="3d")

        # 오른쪽: 오일러각 시계열
        ax_euler = fig.add_subplot(122)

        # 시계열 버퍼
        max_points = 300
        t_buf = []
        roll_buf = []
        pitch_buf = []
        yaw_buf = []
        t_start = time.monotonic()

        hz_time = time.monotonic()
        hz_count = 0

        try:
            while self._running:
                with self.lock:
                    quat = self.quat.copy()
                    euler = self.euler_deg.copy()
                    gyro = self.gyro.copy()
                    accel = self.accel.copy()
                    count = self.sample_count

                roll_d, pitch_d, yaw_d = euler

                # Hz 계산
                now = time.monotonic()
                if now - hz_time >= 1.0:
                    self.hz = (count - hz_count) / (now - hz_time)
                    hz_count = count
                    hz_time = now

                # ── 3D 박스 업데이트 ──
                ax3d.cla()
                R = rotation_matrix_from_quat(quat)
                rotated = (R @ VERTICES.T).T

                polys = [[rotated[v] for v in face] for face in FACES]
                collection = Poly3DCollection(polys, linewidths=1.5, edgecolors="k")
                collection.set_facecolors(FACE_COLORS)
                ax3d.add_collection3d(collection)

                # 축 표시 (회전된 좌표축)
                origin = np.array([0, 0, 0])
                axis_len = 1.2
                colors = ["r", "g", "b"]
                labels = ["X", "Y", "Z"]
                for i in range(3):
                    axis = R[:, i] * axis_len
                    ax3d.quiver(*origin, *axis, color=colors[i],
                                arrow_length_ratio=0.1, linewidth=2)
                    ax3d.text(*(axis * 1.15), labels[i], color=colors[i],
                              fontsize=10, fontweight="bold")

                # 정면 방향 화살표 (파란 렌즈 방향)
                front = R @ np.array([0, 0, 1]) * 1.0
                ax3d.quiver(*origin, *front, color="blue",
                            arrow_length_ratio=0.15, linewidth=2.5,
                            linestyle="--", alpha=0.6)

                lim = 1.5
                ax3d.set_xlim(-lim, lim)
                ax3d.set_ylim(-lim, lim)
                ax3d.set_zlim(-lim, lim)
                ax3d.set_xlabel("X")
                ax3d.set_ylabel("Y")
                ax3d.set_zlabel("Z")
                ax3d.set_title(
                    f"Roll(tilt): {roll_d:+6.1f}°  "
                    f"Pitch(nod): {pitch_d:+6.1f}°  "
                    f"Yaw(turn): {yaw_d:+6.1f}°\n"
                    f"Gyro: ({gyro[0]:+.3f}, {gyro[1]:+.3f}, {gyro[2]:+.3f}) rad/s\n"
                    f"Accel: ({accel[0]:+.2f}, {accel[1]:+.2f}, {accel[2]:+.2f}) m/s²  "
                    f"|a|={np.linalg.norm(accel):.2f}\n"
                    f"{self.hz:.0f} Hz  [R=reset]",
                    fontsize=9, family="monospace",
                )
                ax3d.view_init(elev=20, azim=-60)

                # ── 오일러각 시계열 ──
                elapsed = now - t_start
                t_buf.append(elapsed)
                roll_buf.append(roll_d)
                pitch_buf.append(pitch_d)
                yaw_buf.append(yaw_d)

                # 버퍼 제한
                if len(t_buf) > max_points:
                    t_buf[:] = t_buf[-max_points:]
                    roll_buf[:] = roll_buf[-max_points:]
                    pitch_buf[:] = pitch_buf[-max_points:]
                    yaw_buf[:] = yaw_buf[-max_points:]

                ax_euler.cla()
                ax_euler.plot(t_buf, roll_buf, "r-", linewidth=1.2, label="Roll (tilt)")
                ax_euler.plot(t_buf, pitch_buf, "g-", linewidth=1.2, label="Pitch (nod)")
                ax_euler.plot(t_buf, yaw_buf, "b-", linewidth=1.2, label="Yaw (turn)")
                ax_euler.set_xlabel("Time (s)")
                ax_euler.set_ylabel("Angle (°)")
                ax_euler.set_title("Euler Angles Over Time")
                ax_euler.legend(loc="upper right")
                ax_euler.grid(True, alpha=0.3)
                if len(t_buf) > 1:
                    ax_euler.set_xlim(max(0, t_buf[-1] - 15), t_buf[-1] + 0.5)
                ax_euler.set_ylim(-180, 180)

                fig.tight_layout()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.03)  # ~30fps

        except KeyboardInterrupt:
            print("\n[OK] Visualization stopped.")
        finally:
            imu.stop()
            time.sleep(0.2)
            imu.disconnect()
            plt.close("all")


if __name__ == "__main__":
    viz = IMUVisualizer()
    viz.run()
