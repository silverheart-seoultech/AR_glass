"""
XREAL Air IMU 사용 예제

동료가 IMU 값을 받아서 Pan-Tilt 모터 등에 연결할 때 참고하는 코드.

실행 전:
    1. sudo cp 99-xreal-air.rules /etc/udev/rules.d/
    2. sudo udevadm control --reload-rules && sudo udevadm trigger
    3. 글래스 USB-C 연결 (뽑았다가 다시 꽂기)
    4. pip install numpy
"""

from xreal_imu import XREALAirIMU


# ─── 예제 1: 가장 간단한 실행 ───────────────────────────────
# 터미널에 IMU + Orientation 값을 실시간 출력한다.
# python3 -u xreal_imu.py 와 동일.

def example_basic():
    imu = XREALAirIMU()
    imu.connect()
    imu.stream(enable_fusion=True)  # Ctrl+C로 정지
    imu.disconnect()


# ─── 예제 2: 콜백으로 원하는 값만 꺼내 쓰기 ──────────────────
# Pan-Tilt 모터 제어 등에 연결할 때 이 패턴을 쓰면 된다.

def example_callback():
    import time

    MOTOR_HZ = 50
    last_send = 0

    def on_imu(imu_data, orientation):
        nonlocal last_send
        now = time.monotonic()
        if now - last_send < 1.0 / MOTOR_HZ:
            return  # 50Hz로 throttle
        last_send = now

        roll, pitch, yaw = orientation.euler_deg
        print(f"Pan(Yaw): {yaw:+7.2f}°  Tilt(Pitch): {pitch:+7.2f}°  Roll: {roll:+7.2f}°")

        # 여기에 모터 제어 코드 삽입
        # send_to_motor(pan=yaw, tilt=pitch)

    imu = XREALAirIMU()
    imu.connect()
    imu.stream(callback=on_imu, enable_fusion=True)
    imu.disconnect()


# ─── 예제 3: raw 가속도/자이로만 사용 (센서 퓨전 없이) ────────

def example_raw():
    def on_imu(imu_data, _orientation):
        print(
            f"Gyro: ({imu_data.gyro_x:+.4f}, {imu_data.gyro_y:+.4f}, {imu_data.gyro_z:+.4f}) rad/s  "
            f"Accel: ({imu_data.accel_x:+.3f}, {imu_data.accel_y:+.3f}, {imu_data.accel_z:+.3f}) m/s²"
        )

    imu = XREALAirIMU()
    imu.connect()
    imu.stream(callback=on_imu, enable_fusion=False)
    imu.disconnect()


# ─── 예제 4: 텔레오퍼레이션 에피소드 리셋 ─────────────────────
# 에피소드 시작 시 현재 자세를 0점으로 리셋.
# 이후 모든 orientation은 리셋 시점 기준 상대값.

def example_teleop_reset():
    import threading

    imu = XREALAirIMU()
    imu.connect()
    imu.calibrate_static()

    # 초기 리셋 (현재 자세 = 원점)
    imu.reset_orientation()

    def on_imu(imu_data, orientation):
        # orientation은 리셋 시점 기준 상대값
        roll, pitch, yaw = orientation.euler_deg
        # 여기서 로봇 Pan-Tilt 등에 전달
        pass

    # Enter 키를 누르면 에피소드 리셋
    def reset_thread():
        while True:
            input("Press Enter to reset episode orientation...")
            imu.reset_orientation()

    t = threading.Thread(target=reset_thread, daemon=True)
    t.start()

    imu.stream(callback=on_imu, enable_fusion=True)
    imu.disconnect()


if __name__ == "__main__":
    # 원하는 예제를 골라서 실행
    # example_basic()
    # example_callback()
    # example_raw()
    example_teleop_reset()
