#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # ダークテーマの設定
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
