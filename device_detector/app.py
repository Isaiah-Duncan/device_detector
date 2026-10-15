"""Entry point: python -m device_detector"""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6 import QtWidgets
    from .ui.main_window import STYLE, MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    code = app.exec()
    import pyqtgraph as pg
    from PySide6 import QtCore
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    pg.cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
