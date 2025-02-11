# -*- coding: utf-8 -*-
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QProgressBar,
    QMessageBox, QScrollArea
)
import sys
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon, QPalette, QColor

from .widgets.drop_area import DropArea
from .widgets.model_selector import ModelSelector
from .widgets.format_settings import FormatSettings
from .widgets.settings_tab import SettingsTab
from .utils.config import Config
from .utils.esrgan import ESRGANProcessor
from .widgets.launcher_cell import LauncherCell
from .utils.settings import Settings

class UpscaleThread(QThread):
    """アップスケール処理を実行するスレッド"""
    def __init__(self, processor, *args, **kwargs):
        super().__init__()
        self.processor = processor
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        input_paths = self.args[0]  # 最初の引数がパスのリスト
        if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
            # 単一のディレクトリが選択された場合
            self.processor.process_directory(
                input_paths[0],  # ディレクトリパス
                self.args[1],    # model_name
                self.args[2],    # scale
                self.args[3],    # output_format
                self.args[4],    # jpg_quality
                self.args[5],    # suffix
                self.args[6],    # output_dir
                self.args[7]     # rename_settings
            )
        else:
            # ファイルまたは複数のパスが選択された場合
            self.processor.process_files(
                input_paths,     # ファイルパスのリスト
                self.args[1],    # model_name
                self.args[2],    # scale
                self.args[3],    # output_format
                self.args[4],    # jpg_quality
                self.args[5],    # suffix
                self.args[6],    # output_dir
                self.args[7]     # rename_settings
            )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HkEzLauncher")
        self.setMinimumSize(400, 300)
        
        # メインウィジェットとレイアウトの設定
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # スクロールエリアの設定
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        
        # 設定の読み込み
        self.settings = Settings()
        
        # 初期セルの追加
        if not self.settings.get_cells():
            self.add_cell()
        else:
            for cell_data in self.settings.get_cells():
                self.add_cell(cell_data)
        
        # 追加ボタン
        self.add_button = QPushButton("+")
        self.add_button.clicked.connect(self.add_cell)
        
        # レイアウトの組み立て
        self.layout.addWidget(self.scroll_area)
        self.layout.addWidget(self.add_button)
        
        # スタイルシートの適用
        self.apply_stylesheet()
    
    def add_cell(self, cell_data=None):
        cell = LauncherCell(self, cell_data)
        self.scroll_layout.addWidget(cell)
        self.settings.save_cells()
    
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #3b3b3b;
                color: #ffffff;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
        """)
