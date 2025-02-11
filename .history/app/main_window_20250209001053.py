# -*- coding: utf-8 -*-
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QProgressBar,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon, QPalette, QColor

from .widgets.drop_area import DropArea
from .widgets.model_selector import ModelSelector
from .widgets.format_settings import FormatSettings
from .widgets.settings_tab import SettingsTab
from .utils.config import Config
from .utils.esrgan import ESRGANProcessor

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
        self.setWindowTitle("HkPortaScale")
        self.setWindowIcon(QIcon("icon/app_icon.ico"))
        self.setMinimumSize(800, 600)
        
        # 設定の初期化
        self.config = Config()
        
        # プロセッサーの初期化
        self.processor = ESRGANProcessor(self.config)
        self.processor.progress.connect(self._on_progress)
        self.processor.progress_value.connect(self._on_progress_value)
        self.processor.error.connect(self._on_error)
        self.processor.finished.connect(self._on_finished)
        
        # 処理スレッド
        self.upscale_thread = None
        
        # ダークテーマの設定
        self.setup_dark_theme()
        
        # メインウィジェットとレイアウトの設定
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # タブウィジェットの作成
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_main_tab(), "メイン")
        self.tabs.addTab(self.create_settings_tab(), "詳細設定")
        layout.addWidget(self.tabs)
    
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
        
        # スタイルシートの読み込み
        style_path = os.path.join(os.path.dirname(__file__), "resources", "style.qss")
        with open(style_path, 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())
    
    def create_main_tab(self):
        """メインタブの作成"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # D&D領域
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_area)
        
        # モデル選択
        self.model_selector = ModelSelector(self.config)
        layout.addWidget(self.model_selector)
        
        # フォーマット設定
        self.format_settings = FormatSettings(self.config)
        layout.addWidget(self.format_settings)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # 実行ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.upscale_button = QPushButton("アップスケール実行")
        self.upscale_button.setObjectName("upscaleButton")
        self.upscale_button.setEnabled(False)
        self.upscale_button.clicked.connect(self._on_upscale_clicked)
        button_layout.addWidget(self.upscale_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return widget
    
    def create_settings_tab(self):
        """詳細設定タブの作成"""
        self.settings_tab = SettingsTab(self.config)
        
        # 設定変更時の処理
        self.settings_tab.settings_changed.connect(self.config.update_settings)
        
        return self.settings_tab
    
    def _on_files_dropped(self, paths):
        """ファイルがドロップされた時の処理"""
        if paths:
            self.upscale_button.setEnabled(True)
            if len(paths) == 1:
                self.drop_area.set_status(f"選択: {os.path.basename(paths[0])}")
            else:
                self.drop_area.set_status(f"選択: {len(paths)}個のファイル")
            self.input_paths = paths
    
    def _on_upscale_clicked(self):
        """アップスケール実行ボタンがクリックされた時の処理"""
        if not hasattr(self, 'input_paths') or not self.input_paths:
            return
        
        # UIの更新
        self.upscale_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)  # 0-100%の範囲を設定
        self.progress_bar.setValue(0)        # 初期値を0に設定
        self.progress_bar.show()
        
        # 設定の取得
        format_settings = self.format_settings.get_settings()
        output_settings = self.settings_tab.get_settings()
        
        # 出力ディレクトリの設定
        if len(self.input_paths) > 1:
            # 複数ファイルの場合、設定されたディレクトリを作成
            first_file = self.input_paths[0]
            first_file_dir = os.path.dirname(first_file)
            output_dir = os.path.join(first_file_dir, output_settings["output_dir"])
        else:
            output_dir = None

        # 処理スレッドの作成と開始
        self.upscale_thread = UpscaleThread(
            self.processor,
            self.input_paths,
            self.model_selector.get_selected_model(),
            format_settings["scale"],
            format_settings["format"],
            format_settings["jpg_quality"],
            output_settings["output_suffix"],
            output_dir,
            output_settings     # リネーム設定を含む
        )
        self.upscale_thread.finished.connect(self._on_thread_finished)
        self.upscale_thread.start()
    
    def _on_progress(self, message):
        """進捗メッセージの更新"""
        self.progress_bar.setFormat(f"%p% - {message}")
    
    def _on_progress_value(self, value):
        """進捗値の更新"""
        self.progress_bar.setValue(value)
    
    def _on_error(self, message):
        """エラー発生時の処理"""
        QMessageBox.warning(self, "エラー", message)
    
    def _on_finished(self):
        """処理完了時の処理"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
    
    def _on_thread_finished(self):
        """スレッド完了時の処理"""
        self.upscale_button.setEnabled(True)
        self.upscale_thread = None
