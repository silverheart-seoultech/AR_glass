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


# ── 글래스 형태의 3D 꼭짓점 (안경 모양) ──────────────────────────
# 가로 넓고, 세로 낮고, 앞뒤 얇은 박스
W, H, D = 1.6, 0.5, 0.3  # width, height, depth

VERTICES = np.array([
    [-W/2, -H/2, -D/2],  # 0: left-bottom-back
    [+W/2, -H/2, -D/2],  # 1: right-bottom-back
    [+W/2, +H/2, -D/2],  # 2: right-top-back
    [-W/2, +H/2, -D/2],  # 3: left-top-back
    [-W/2, -H/2, +D/2],  # 4: left-bottom-front
    [+W/2, -H/2, +D/2],  # 5: right-bottom-front
    [+W/2, +H/2, +D/2],  # 6: right-top-front
    [-W/2, +H/2, +D/2],  # 7: left-top-front
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


def rotation_matrix(roll, pitch, yaw):
    """오일러각(rad) → 3x3 회전 행렬."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


class IMUVisualizer:
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro = np.zeros(3)
        self.accel = np.zeros(3)
        self.sample_count = 0
        self.hz = 0.0
        self.lock = threading.Lock()
        self._running = True

    def imu_callback(self, imu_data: IMUData, orientation: Orientation):
        with self.lock:
            self.roll = np.radians(orientation.euler_deg[0])
            self.pitch = np.radians(orientation.euler_deg[1])
            self.yaw = np.radians(orientation.euler_deg[2])
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
                    roll = self.roll
                    pitch = self.pitch
                    yaw = self.yaw
                    gyro = self.gyro.copy()
                    accel = self.accel.copy()
                    count = self.sample_count

                # Hz 계산
                now = time.monotonic()
                if now - hz_time >= 1.0:
                    self.hz = (count - hz_count) / (now - hz_time)
                    hz_count = count
                    hz_time = now

                # ── 3D 박스 업데이트 ──
                ax3d.cla()
                R = rotation_matrix(roll, pitch, yaw)
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
                    f"Roll: {np.degrees(roll):+6.1f}°  "
                    f"Pitch: {np.degrees(pitch):+6.1f}°  "
                    f"Yaw: {np.degrees(yaw):+6.1f}°\n"
                    f"Gyro: ({gyro[0]:+.3f}, {gyro[1]:+.3f}, {gyro[2]:+.3f}) rad/s\n"
                    f"Accel: ({accel[0]:+.2f}, {accel[1]:+.2f}, {accel[2]:+.2f}) m/s²  "
                    f"|a|={np.linalg.norm(accel):.2f}\n"
                    f"{self.hz:.0f} Hz",
                    fontsize=9, family="monospace",
                )
                ax3d.view_init(elev=20, azim=-60)

                # ── 오일러각 시계열 ──
                elapsed = now - t_start
                t_buf.append(elapsed)
                roll_buf.append(np.degrees(roll))
                pitch_buf.append(np.degrees(pitch))
                yaw_buf.append(np.degrees(yaw))

                # 버퍼 제한
                if len(t_buf) > max_points:
                    t_buf[:] = t_buf[-max_points:]
                    roll_buf[:] = roll_buf[-max_points:]
                    pitch_buf[:] = pitch_buf[-max_points:]
                    yaw_buf[:] = yaw_buf[-max_points:]

                ax_euler.cla()
                ax_euler.plot(t_buf, roll_buf, "r-", linewidth=1.2, label="Roll")
                ax_euler.plot(t_buf, pitch_buf, "g-", linewidth=1.2, label="Pitch")
                ax_euler.plot(t_buf, yaw_buf, "b-", linewidth=1.2, label="Yaw")
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
