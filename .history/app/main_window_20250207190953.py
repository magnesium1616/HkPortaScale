# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QRadioButton,
    QPushButton, QTabWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPalette, QColor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HkPortaScale")
        self.setWindowIcon(QIcon("icon/app_icon.ico"))
        self.setMinimumSize(800, 600)
        
        # ダークテーマの設定
        self.setup_dark_theme()
        
        # メインウィジェットとレイアウトの設定
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # タブウィジェットの作成
        tabs = QTabWidget()
        tabs.addTab(self.create_main_tab(), "メイン")
        tabs.addTab(self.create_settings_tab(), "詳細設定")
        layout.addWidget(tabs)
    
    def setup_dark_theme(self):
        """ダークテーマの設定"""
        palette = QPalette()
        
        # ウィンドウの背景色
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        
        # ウィジェットの背景色
        palette.setColor(QPalette.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        
        # テキストの色
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.ButtonText, Qt.white)
        
        # ボタンの色
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        
        # 選択項目の色
        palette.setColor(QPalette.Highlight, QColor(255, 140, 0))  # オレンジ色
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(palette)
    
    def create_main_tab(self):
        """メインタブの作成"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ここに D&D 領域、モデル選択、スケール設定などを追加予定
        layout.addWidget(QLabel("メインタブ（実装予定）"))
        
        return widget
    
    def create_settings_tab(self):
        """詳細設定タブの作成"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ここに出力設定、サフィックス設定などを追加予定
        layout.addWidget(QLabel("詳細設定タブ（実装予定）"))
        
        return widget
