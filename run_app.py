"""Entry script used by PyInstaller and for `python run_app.py`."""
from device_detector.app import main

if __name__ == "__main__":
    raise SystemExit(main())
