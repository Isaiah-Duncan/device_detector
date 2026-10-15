# PyInstaller spec. Build with: pyinstaller device_detector.spec --noconfirm --clean
from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("bleak") + ["bleak.backends.winrt", "pyqtgraph"]
a = Analysis(["run_app.py"], pathex=["."], hiddenimports=hidden, excludes=["tkinter", "matplotlib"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DeviceDetector", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DeviceDetector")
