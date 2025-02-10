import sys
import os
import torch
from torch import nn
import torch.onnx
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt

class RRDB(nn.Module):
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        self.rdb1 = RDB(nf)
        self.rdb2 = RDB(nf)
        self.rdb3 = RDB(nf)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RDB(nn.Module):
    def __init__(self, nf=64):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = self.conv5(x4)
        return x5 * 0.2 + x

class RealESRGANModel(nn.Module):
    def __init__(self, num_rrdb=23):
        super(RealESRGANModel, self).__init__()
        
        # 入力層
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.ModuleList([RRDB(64) for _ in range(num_rrdb)])
        
        # 特徴抽出用の畳み込み層
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        
        # アップサンプリング層
        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 出力層
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 初期特徴抽出
        feat = self.conv_first(x)
        body_feat = feat.clone()
        
        # RRDB blocks
        for block in self.body:
            body_feat = block(body_feat)
        
        # グローバルな特徴結合
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # アップサンプリング
        feat = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle(self.upconv2(feat)))
        
        # 最終出力
        feat = self.lrelu(self.conv_hr(feat))
        feat = self.conv_last(feat)
        
        return feat

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
            
        # デバッグ情報の出力
        print("Loading state dict from:", model_path)
        print("State dict keys:")
        for key in state_dict.keys():
            print(f"- {key}")
        
        return state_dict

    def convert_model(self):
        if not self.selected_file:
            return
        
        try:
            self.status_label.setText("変換中...")
            self.convert_button.setEnabled(False)
            self.select_button.setEnabled(False)
            
            # モデルの読み込みと変換
            model = RealESRGANModel()
            state_dict = self.load_state_dict(self.selected_file)
            
            # モデルの状態をデバッグ出力
            print("\nModel state dict keys:")
            model_state = model.state_dict()
            for key in model_state.keys():
                print(f"- {key}")
            
            model.load_state_dict(state_dict)
            model.eval()
            
            # 出力パスの生成
            output_path = os.path.splitext(self.selected_file)[0] + '.onnx'
            
            # ダミー入力の作成（入力サイズを64x64に設定）
            dummy_input = torch.randn(1, 3, 64, 64)
            
            # ONNXへの変換
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=11,
                verbose=True
            )
            
            self.status_label.setText(f"変換完了: {os.path.basename(output_path)}")
            QMessageBox.information(self, "成功", "モデルの変換が完了しました。")
            
        except Exception as e:
            self.status_label.setText("エラーが発生しました")
            error_msg = f"変換中にエラーが発生しました:\n{str(e)}"
            print(f"Error details: {str(e)}")
            QMessageBox.critical(self, "エラー", error_msg)
        
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
