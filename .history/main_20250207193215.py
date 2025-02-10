#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.main_window import MainWindow

def load_stylesheet(path: str) -> str:
    """スタイルシートを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"スタイルシートの読み込みに失敗しました: {e}")
        return ""

def main():
    app = QApplication(sys.argv)
    
    # High DPI対応（Qt6の新しい方法）
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # ダークテーマの設定
    app.setStyle('Fusion')
    
    # スタイルシートの適用
    style_path = os.path.join('app', 'resources', 'style.qss')
    app.setStyleSheet(load_stylesheet(style_path))
    
    # メインウィンドウの作成と表示
    window = MainWindow()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
