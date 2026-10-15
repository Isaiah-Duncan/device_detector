# PyInstaller spec for the Sensor Lab command-line program.
# Build with: pyinstaller sensor_lab.spec --noconfirm --clean
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden = collect_submodules("bleak") + ["bleak.backends.winrt", "sounddevice", "psutil"]
datas = collect_data_files("sounddevice")
a = Analysis(["run_sensor_lab.py"], pathex=["."], hiddenimports=hidden, datas=datas, excludes=["tkinter", "matplotlib", "PySide6"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SensorLab", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="SensorLab")
