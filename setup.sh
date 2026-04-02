#!/bin/bash
# XREAL Air (NR-7100RGL) Environment Setup Script

set -e

echo "=== XREAL Air IMU Setup ==="
echo ""

# 1. Install system dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq libhidapi-hidraw0 libhidapi-dev python3-pip python3-venv

# 2. Install udev rules
echo "[2/4] Installing udev rules..."
sudo cp 99-xreal-air.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  udev rules installed for XREAL Air (VID:3318, PID:0424)"

# 3. Create venv and install Python packages
echo "[3/4] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Verify
echo "[4/4] Verifying setup..."
python3 -c "import hid; print('  hidapi OK')"
python3 -c "import numpy; print('  numpy OK')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Connect XREAL Air glasses via USB-C"
echo "  2. Verify with: lsusb | grep 3318"
echo "  3. Run: source venv/bin/activate && python3 xreal_imu.py"
