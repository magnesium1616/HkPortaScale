# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QSlider, QComboBox,
    QGroupBox
)
from PySide6.QtCore import Signal, Qt

class FormatSettings(QWidget):
    """スケール倍率と出力フォーマット設定ウィジェット"""
    
    # 設定変更時のシグナル
    settings_changed = Signal(dict)  # 設定値の辞書
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
    
    def setup_ui(self):
        """UIの初期設定"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # スケール倍率設定
        scale_group = QGroupBox("スケール倍率")
        scale_layout = QHBoxLayout()
        
        self.scale_group = QButtonGroup(self)
        scales = [("2x", 2), ("3x", 3), ("4x", 4)]
        for text, value in scales:
            radio = QRadioButton(text)
            radio.scale_value = value
            self.scale_group.addButton(radio)
            scale_layout.addWidget(radio)
            if value == 4:  # デフォルトは4x
                radio.setChecked(True)
        
        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)
        
        # 出力フォーマット設定
        format_group = QGroupBox("出力フォーマット")
        format_layout = QVBoxLayout()
        
        # フォーマット選択
        format_hlayout = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG"])
        # デフォルト値を設定
        self.format_combo.setCurrentText(self.config.get_setting("default_format").upper())
        format_hlayout.addWidget(QLabel("フォーマット:"))
        format_hlayout.addWidget(self.format_combo)
        format_layout.addLayout(format_hlayout)
        
        # JPG品質設定
        quality_layout = QHBoxLayout()
        self.quality_label = QLabel("JPG品質: 80%")
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(self.config.get_setting("jpg_quality"))
        quality_layout.addWidget(self.quality_label)
        quality_layout.addWidget(self.quality_slider)
        format_layout.addLayout(quality_layout)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # シグナル接続
        self.scale_group.buttonClicked.connect(self._emit_settings)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
    
    def _on_format_changed(self, format_text: str):
        """フォーマット選択が変更された時"""
        self.quality_slider.setEnabled(format_text == "JPG")
        self._emit_settings()
    
    def _on_quality_changed(self, value: int):
        """JPG品質が変更された時"""
        self.quality_label.setText(f"JPG品質: {value}%")
        self._emit_settings()
    
    def _get_current_settings(self) -> dict:
        """現在の設定値を取得"""
        return {
            "scale": self.scale_group.checkedButton().scale_value,
            "format": self.format_combo.currentText().lower(),
            "jpg_quality": self.quality_slider.value()
        }
    
    def _emit_settings(self, *args):
        """設定変更を通知"""
        self.settings_changed.emit(self._get_current_settings())
    
    def get_settings(self) -> dict:
        """現在の設定値を取得"""
        return self._get_current_settings()
