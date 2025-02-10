import sys
import os
import torch
from torch import nn
import torch.onnx
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)
    
    def forward(self, x):
        return self.conv(x)

class ModelConverterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("モデル変換ツール")
        self.setFixedSize(400, 200)
        
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setAlignment(Qt.AlignCenter)
        
        # ファイル選択ボタン
        self.select_button = QPushButton("モデルファイルを選択")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        # 選択されたファイルパスを表示するラベル
        self.file_label = QLabel("ファイルが選択されていません")
        self.file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_label)
        
        # 変換ボタン
        self.convert_button = QPushButton("変換開始")
        self.convert_button.clicked.connect(self.convert_model)
        self.convert_button.setEnabled(False)
        layout.addWidget(self.convert_button)
        
        # ステータスラベル
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.selected_file = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "モデルファイルを選択",
            "",
            "PyTorchモデル (*.pth *.pt)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.convert_button.setEnabled(True)
            self.status_label.setText("")

    def load_state_dict(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        return state_dict

    def convert_model(self):
        if not self.selected_file:
            return
        
        try:
            self.status_label.setText("変換中...")
            self.convert_button.setEnabled(False)
            self.select_button.setEnabled(False)
            
            # モデルの読み込みと変換
            model = SimpleModel()
            state_dict = self.load_state_dict(self.selected_file)
            model.load_state_dict(state_dict)
            model.eval()
            
            # 出力パスの生成
            output_path = os.path.splitext(self.selected_file)[0] + '.onnx'
            
            # ダミー入力の作成
            dummy_input = torch.randn(1, 3, 64, 64)
            
            # ONNXへの変換
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=11
            )
            
            self.status_label.setText(f"変換完了: {os.path.basename(output_path)}")
            QMessageBox.information(self, "成功", "モデルの変換が完了しました。")
            
        except Exception as e:
            self.status_label.setText("エラーが発生しました")
            QMessageBox.critical(self, "エラー", f"変換中にエラーが発生しました: {str(e)}")
        
        finally:
            self.convert_button.setEnabled(True)
            self.select_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ModelConverterWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
