# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QSlider, QComboBox,
    QGroupBox, QDoubleSpinBox
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
            if value == 4 and not self.config.get_setting("custom_scale_enabled"):  # カスタムが無効の場合のデフォルト
                radio.setChecked(True)
        
        # カスタム倍率オプション
        self.custom_radio = QRadioButton("カスタム")
        self.custom_radio.scale_value = None  # カスタム値はスピンボックスから取得
        self.scale_group.addButton(self.custom_radio)
        scale_layout.addWidget(self.custom_radio)
        
        self.custom_scale = QDoubleSpinBox()
        self.custom_scale.setRange(0.1, 100.0)  # 広めに設定
        self.custom_scale.setDecimals(2)  # 小数点以下2桁
        self.custom_scale.setSingleStep(0.1)
        self.custom_scale.setValue(self.config.get_setting("custom_scale_value"))
        self.custom_scale.setEnabled(False)  # デフォルトは無効
        scale_layout.addWidget(self.custom_scale)
        
        # カスタムモードが有効な場合、カスタムラジオボタンを選択
        if self.config.get_setting("custom_scale_enabled"):
            self.custom_radio.setChecked(True)
            self.custom_scale.setEnabled(True)
        
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
        jpg_quality = self.config.get_setting("jpg_quality")
        self.quality_label = QLabel(f"JPG品質: {jpg_quality}%")
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(self.config.get_setting("jpg_quality"))
        quality_layout.addWidget(self.quality_label)
        quality_layout.addWidget(self.quality_slider)
        format_layout.addLayout(quality_layout)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # シグナル接続
        self.scale_group.buttonClicked.connect(self._on_scale_changed)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
        self.custom_scale.valueChanged.connect(self._emit_settings)
    
    def _on_format_changed(self, format_text: str):
        """フォーマット選択が変更された時"""
        is_jpg = format_text == "JPG"
        self.quality_slider.setEnabled(is_jpg)
        self.quality_label.setEnabled(is_jpg)  # ラベルも連動して有効/無効を切り替え
        self._emit_settings()
        
    def _on_quality_changed(self, value: int):
        """JPG品質が変更された時"""
        self.quality_label.setText(f"JPG品質: {value}%")
        self._emit_settings()
    
    def _on_scale_changed(self, button: QRadioButton):
        """スケール設定が変更された時"""
        is_custom = button == self.custom_radio
        self.custom_scale.setEnabled(is_custom)
        self._emit_settings()
    
    def _get_current_settings(self) -> dict:
        """現在の設定値を取得"""
        checked_button = self.scale_group.checkedButton()
        is_custom = checked_button == self.custom_radio
        
        settings = {
            "scale": self.custom_scale.value() if is_custom else checked_button.scale_value,
            "format": self.format_combo.currentText().lower(),
            "jpg_quality": self.quality_slider.value(),
            "custom_scale_enabled": is_custom,
            "custom_scale_value": self.custom_scale.value()
        }
        return settings
    
    def _emit_settings(self, *args):
        """設定変更を通知"""
        self.settings_changed.emit(self._get_current_settings())
    
    def get_settings(self) -> dict:
        """現在の設定値を取得"""
        return self._get_current_settings()
