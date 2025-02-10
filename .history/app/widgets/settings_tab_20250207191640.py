# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QGroupBox
)
from PySide6.QtCore import Signal

class SettingsTab(QWidget):
    """詳細設定タブのウィジェット"""
    
    # 設定変更時のシグナル
    settings_changed = Signal(dict)  # 設定値の辞書
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """UIの初期設定"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # 出力ファイル設定
        file_group = QGroupBox("出力ファイル設定")
        file_layout = QVBoxLayout()
        
        # サフィックス設定
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("出力ファイルのサフィックス:"))
        self.suffix_edit = QLineEdit("_upscaled")
        self.suffix_edit.setPlaceholderText("例: _upscaled")
        self.suffix_edit.textChanged.connect(self._emit_settings)
        suffix_layout.addWidget(self.suffix_edit)
        file_layout.addLayout(suffix_layout)
        
        # サフィックスの説明
        suffix_desc = QLabel(
            "※ 出力ファイル名は「元のファイル名 + サフィックス + 拡張子」となります\n"
            "   例: image.jpg → image_upscaled.jpg"
        )
        suffix_desc.setStyleSheet("color: #999;")
        file_layout.addWidget(suffix_desc)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 出力ディレクトリ設定
        dir_group = QGroupBox("出力ディレクトリ設定")
        dir_layout = QVBoxLayout()
        
        # ディレクトリ名設定
        dir_name_layout = QHBoxLayout()
        dir_name_layout.addWidget(QLabel("出力ディレクトリ名:"))
        self.dir_name_edit = QLineEdit("upscale")
        self.dir_name_edit.setPlaceholderText("例: upscale")
        self.dir_name_edit.textChanged.connect(self._emit_settings)
        dir_name_layout.addWidget(self.dir_name_edit)
        dir_layout.addLayout(dir_name_layout)
        
        # ディレクトリの説明
        dir_desc = QLabel(
            "※ 複数ファイルまたはフォルダを処理する場合、\n"
            "   このディレクトリが作成され、その中に出力ファイルが保存されます"
        )
        dir_desc.setStyleSheet("color: #999;")
        dir_layout.addWidget(dir_desc)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 余白を下に追加
        layout.addStretch()
    
    def _emit_settings(self):
        """設定変更を通知"""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def get_settings(self) -> dict:
        """現在の設定値を取得"""
        return {
            "output_suffix": self.suffix_edit.text(),
            "output_dir": self.dir_name_edit.text()
        }
    
    def set_settings(self, settings: dict):
        """設定値を設定"""
        if "output_suffix" in settings:
            self.suffix_edit.setText(settings["output_suffix"])
        if "output_dir" in settings:
            self.dir_name_edit.setText(settings["output_dir"])
