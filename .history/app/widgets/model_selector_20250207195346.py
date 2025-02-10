# -*- coding: utf-8 -*-
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox
)
from PySide6.QtCore import Signal

class ModelSelector(QWidget):
    """モデル選択ウィジェット"""
    
    # モデルが選択された時のシグナル
    model_selected = Signal(str)  # モデル名
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        """UIの初期設定"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # ラベル
        label = QLabel("アップスケールモデル")
        layout.addWidget(label)
        
        # コンボボックス
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.combo)
    
    def load_models(self):
        """models/ディレクトリからモデルを読み込む"""
        models_dir = "models"
        # モデル名とファイル名のマッピング
        model_files = {
            "realesr-animevideov3": "realesr-animevideov3.pth",
            "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B.pth",
            "realesrgan-x4plus": "RealESRGAN_x4plus.pth",
            "4xUltrasharp": "4xUltrasharp_4xUltrasharpV10.pt"
        }
        
        # モデル名と表示名のマッピング
        model_display_names = {
            "realesr-animevideov3": "アニメ動画 V3",
            "realesrgan-x4plus-anime": "アニメ画像 4x",
            "realesrgan-x4plus": "写真 4x",
            "4xUltrasharp": "Ultrasharp 4x"
        }
        
        # コンボボックスをクリア
        self.combo.clear()
        
        # 利用可能なモデルを追加
        for model_name, filename in model_files.items():
            if os.path.exists(os.path.join(models_dir, filename)):
                display_name = model_display_names.get(model_name, model_name)
                self.combo.addItem(display_name, model_name)  # 表示名とモデル名を別々に設定
        
        # デフォルトモデルを選択
        default_model = "realesr-animevideov3"
        default_display_name = model_display_names.get(default_model)
        index = self.combo.findText(default_display_name)
        if index >= 0:
            self.combo.setCurrentIndex(index)
    
    def _on_model_changed(self, model_name: str):
        """モデル選択が変更された時"""
        self.model_selected.emit(model_name)
    
    def get_selected_model(self) -> str:
        """現在選択されているモデル名を取得"""
        return self.combo.currentData()  # モデル名を取得
