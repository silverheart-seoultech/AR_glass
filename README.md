# XREAL Air (NR-7100RGL) IMU Reader

XREAL Air AR 글래스에서 USB HID를 통해 IMU(가속도계 + 자이로스코프) 데이터를 수신하고, Madgwick 센서 퓨전으로 Head Orientation(Roll/Pitch/Yaw)을 실시간 계산하는 Python 코드입니다.

## Quick Start

```bash
# 1. udev 규칙 설치 (최초 1회)
sudo cp 99-xreal-air.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 글래스를 뽑았다가 다시 꽂기

# 2. 의존성 설치
pip install numpy

# 3. 실행 (글래스 USB-C 연결 상태에서)
python3 -u xreal_imu.py
```

## 출력 데이터

| 필드 | 단위 | 설명 |
|---|---|---|
| `gyro_x/y/z` | rad/s | 각속도 (자이로스코프) |
| `accel_x/y/z` | m/s² | 선형 가속도 |
| `timestamp_us` | μs | 디바이스 타임스탬프 |
| `euler_deg[0]` Roll | ° | 좌우 기울임 |
| `euler_deg[1]` Pitch | ° | 위아래 끄덕임 (→ Tilt 모터) |
| `euler_deg[2]` Yaw | ° | 좌우 돌림 (→ Pan 모터) |

## 콜백으로 데이터 사용하기

```python
from xreal_imu import XREALAirIMU

def on_imu(imu_data, orientation):
    yaw   = orientation.euler_deg[2]   # Pan
    pitch = orientation.euler_deg[1]   # Tilt
    # send_to_motor(pan=yaw, tilt=pitch)

imu = XREALAirIMU()
imu.connect()
imu.stream(callback=on_imu, enable_fusion=True)
imu.disconnect()
```

## 3D 시각화로 검증

```bash
pip install matplotlib
python3 -u visualize_imu.py
```

- 왼쪽: 글래스 방향을 따라 회전하는 3D 박스 (파란면 = 렌즈 방향)
- 오른쪽: Roll/Pitch/Yaw 시계열 그래프

검증 포인트:
- 좌우 돌리기 → Yaw 변화, 끄덕이기 → Pitch 변화, 기울이기 → Roll 변화
- 정지 시 `|a|` ≈ 9.8 m/s² 이면 가속도계 정상

## 센서 퓨전 Warmup

시작 직후 ~0.5초간 Madgwick 필터가 높은 보정 강도(β=2.5→0.1)로 현재 자세에 빠르게 수렴합니다.
이후에는 안정적인 추적 모드(β=0.1)로 전환됩니다.
글래스를 **정면을 바라본 상태**에서 시작하면 초기 기준점이 정확합니다.

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| "XREAL Air not found" | `lsusb \| grep 3318` 확인, udev 규칙 설치 후 재연결 |
| "Permission denied" | `sudo cp 99-xreal-air.rules /etc/udev/rules.d/` 후 재연결 |
| 0 samples / 데이터 안 나옴 | **글래스를 USB에서 뽑았다가 다시 꽂기** (디바이스 리셋) |
| hidapi "open failed" | `pip install hidapi` 사용 금지 — 코드가 /dev/hidraw 직접 접근 방식 사용 |

## 파일 구성

| 파일 | 설명 |
|---|---|
| `xreal_imu.py` | IMU 수신 + Madgwick 센서 퓨전 메인 코드 |
| `visualize_imu.py` | 3D orientation 시각화 (matplotlib) |
| `example_usage.py` | 콜백 사용 예제 (Pan-Tilt 모터 연동 등) |
| `99-xreal-air.rules` | udev 권한 규칙 |
| `requirements.txt` | Python 의존성 (numpy) |
| `setup.sh` | 환경 자동 셋업 스크립트 |

## 레퍼런스

- [ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs) — Rust XREAL 드라이버 (프로토콜 레퍼런스)
- [nrealAirLinuxDriver](https://gitlab.com/TheJackiMonster/nrealAirLinuxDriver) — C 기반 Linux 드라이버
- 내부 센서: InvenSense ICM-42688-P (6축 IMU)
- USB: VID `0x3318`, PID `0x0424`, Interface 3 (HID)
