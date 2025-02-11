# -*- coding: utf-8 -*-
import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox
)
from PySide6.QtCore import Signal

class ModelSelector(QWidget):
    """モデル選択ウィジェット"""
    
    # モデルが選択された時のシグナル
    model_selected = Signal(str)  # モデル名
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
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
        models_dir = self.config.get_models_dir()
        
        # コンボボックスをクリア
        self.combo.clear()
        
        try:
            # モデルファイルの検索（.binファイル）
            model_files = []
            for file in os.listdir(models_dir):
                if file.endswith('.bin'):
                    # .binと.paramの両方が存在することを確認
                    model_base = os.path.splitext(file)[0]
                    param_file = model_base + ".param"
                    if os.path.exists(os.path.join(models_dir, param_file)):
                        model_files.append(model_base)
            
            # モデルを追加
            for model_name in sorted(model_files):
                self.combo.addItem(model_name)
            
            # デフォルトモデルを選択
            default_model = "realesrgan-x4plus-anime"
            default_index = self.combo.findText(default_model)
            if default_index >= 0:
                self.combo.setCurrentIndex(default_index)
            else:
                # デフォルトモデルが見つからない場合は最初のモデルを選択
                if self.combo.count() > 0:
                    self.combo.setCurrentIndex(0)
                
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            # エラーが発生した場合でもUIは表示できるようにする
            self.combo.addItem("モデルが見つかりません")
            self.combo.setEnabled(False)
    
    def _on_model_changed(self, model_name: str):
        """モデル選択が変更された時"""
        self.model_selected.emit(model_name)
    
    def get_selected_model(self) -> str:
        """現在選択されているモデル名を取得"""
        return self.combo.currentText()
