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
        # 実行ファイルのパスを取得
        if hasattr(sys, '_MEIPASS'):  # PyInstallerでビルドされた場合
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        models_dir = os.path.join(base_path, "models")
        
        # コンボボックスをクリア
        self.combo.clear()
        
        try:
            # モデルファイルの検索（.pthと.ptファイル）
            model_files = []
            for file in os.listdir(models_dir):
                if file.endswith(('.pth', '.pt')):
                    model_name = os.path.splitext(file)[0]
                    model_files.append(model_name)
            
            # モデルを追加
            for model_name in sorted(model_files):
                self.combo.addItem(model_name)
            
            # デフォルトモデルを選択（最初のモデル）
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
