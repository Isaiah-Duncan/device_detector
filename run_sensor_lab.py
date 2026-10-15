"""Entry script for PyInstaller and for `python run_sensor_lab.py`."""
from device_detector.sensor_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
