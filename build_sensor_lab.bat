@echo off
REM Build the Sensor Lab program. Run from this folder in a Command Prompt.
if not exist .venv ( py -3.11 -m venv .venv || goto :err )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
python -m pytest -q tests || goto :err
pyinstaller sensor_lab.spec --noconfirm --clean || goto :err
echo.
echo Done. Start dist\SensorLab\SensorLab.exe from a Command Prompt.
exit /b 0
:err
echo Build failed. Read the messages above.
exit /b 1
