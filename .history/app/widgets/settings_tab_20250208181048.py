# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QGroupBox, QCheckBox, QSpinBox,
    QPushButton
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
        
        # リネーム設定
        rename_group = QGroupBox("リネーム設定")
        rename_layout = QVBoxLayout()
        
        # リネーム有効化
        self.rename_enabled = QCheckBox("リネーム機能を有効にする")
        self.rename_enabled.stateChanged.connect(self._emit_settings)
        rename_layout.addWidget(self.rename_enabled)
        
        # リネーム名設定
        rename_name_layout = QHBoxLayout()
        rename_name_layout.addWidget(QLabel("リネーム名:"))
        self.rename_name_edit = QLineEdit()
        self.rename_name_edit.setPlaceholderText("例: image")
        self.rename_name_edit.textChanged.connect(self._emit_settings)
        rename_name_layout.addWidget(self.rename_name_edit)
        rename_layout.addLayout(rename_name_layout)
        
        # パディング数設定
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("番号のパディング桁数:"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(1, 10)
        self.padding_spin.setValue(4)
        self.padding_spin.valueChanged.connect(self._emit_settings)
        padding_layout.addWidget(self.padding_spin)
        rename_layout.addLayout(padding_layout)
        
        # リネームの説明
        rename_desc = QLabel(
            "※ リネーム有効時は「リネーム名 + 連番 + サフィックス + 拡張子」となります\n"
            "   例: image0001_upscaled.jpg"
        )
        rename_desc.setStyleSheet("color: #999;")
        rename_layout.addWidget(rename_desc)
        
        rename_group.setLayout(rename_layout)
        layout.addWidget(rename_group)
        
        # リセットボタン
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        self.reset_button = QPushButton("設定をリセット")
        self.reset_button.clicked.connect(self._reset_settings)
        reset_layout.addWidget(self.reset_button)
        layout.addLayout(reset_layout)
        
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
            "output_dir": self.dir_name_edit.text(),
            "rename_enabled": self.rename_enabled.isChecked(),
            "rename_name": self.rename_name_edit.text(),
            "padding_digits": self.padding_spin.value()
        }
    
    def set_settings(self, settings: dict):
        """設定値を設定"""
        if "output_suffix" in settings:
            self.suffix_edit.setText(settings["output_suffix"])
        if "output_dir" in settings:
            self.dir_name_edit.setText(settings["output_dir"])
        if "rename_enabled" in settings:
            self.rename_enabled.setChecked(settings["rename_enabled"])
        if "rename_name" in settings:
            self.rename_name_edit.setText(settings["rename_name"])
        if "padding_digits" in settings:
            self.padding_spin.setValue(settings["padding_digits"])
    
    def _reset_settings(self):
        """設定をデフォルト値にリセット"""
        default_settings = {
            "output_suffix": "_upscaled",
            "output_dir": "upscale",
            "rename_enabled": False,
            "rename_name": "",
            "padding_digits": 4
        }
        self.set_settings(default_settings)
        self._emit_settings()
