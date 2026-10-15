@echo off
REM Build a one-folder Windows program. Run from this folder in a Command Prompt.
py -3.11 -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
python -m pytest -q tests || goto :err
pyinstaller device_detector.spec --noconfirm --clean || goto :err
echo.
echo Done. Start dist\DeviceDetector\DeviceDetector.exe
exit /b 0
:err
echo Build failed. Read the messages above.
exit /b 1
