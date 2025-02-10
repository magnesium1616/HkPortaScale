import sys
import os
import torch
from torch import nn
import torch.onnx
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QCheckBox)
from PySide6.QtCore import Qt
import subprocess

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat.clone()
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super(RRDB, self).__init__()
        self.rdb1 = RDB(num_feat, num_grow_ch)
        self.rdb2 = RDB(num_feat, num_grow_ch)
        self.rdb3 = RDB(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class ModelConverterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("モデル変換ツール")
        self.setFixedSize(400, 300)
        
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setAlignment(Qt.AlignCenter)
        
        # NCNNツールパスの設定
        self.ncnn_path_label = QLabel("NCNN Tools Path:")
        self.ncnn_path_button = QPushButton("NCNNツールパスを選択")
        self.ncnn_path_button.clicked.connect(self.select_ncnn_path)
        layout.addWidget(self.ncnn_path_label)
        layout.addWidget(self.ncnn_path_button)
        
        # ファイル選択ボタン
        self.select_button = QPushButton("モデルファイルを選択")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        # 選択されたファイルパスを表示するラベル
        self.file_label = QLabel("ファイルが選択されていません")
        self.file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_label)

        # params_emaの選択チェックボックス
        self.use_params_ema = QCheckBox("params_emaを使用")
        self.use_params_ema.setChecked(True)
        layout.addWidget(self.use_params_ema)
        
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
        self.ncnn_path = None

    def select_ncnn_path(self):
        ncnn_path = QFileDialog.getExistingDirectory(
            self,
            "NCNNツールのディレクトリを選択",
            ""
        )
        if ncnn_path:
            self.ncnn_path = ncnn_path
            self.ncnn_path_label.setText(f"NCNN Tools Path: {os.path.basename(ncnn_path)}")
            self.check_convert_button()

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
            self.check_convert_button()
            self.status_label.setText("")

    def check_convert_button(self):
        self.convert_button.setEnabled(
            self.selected_file is not None and self.ncnn_path is not None
        )

    def load_state_dict(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        keyname = 'params_ema' if self.use_params_ema.isChecked() else 'params'
        if keyname in state_dict:
            state_dict = state_dict[keyname]
            
        print("Loading state dict from:", model_path)
        print(f"Using key: {keyname}")
        print("State dict keys:")
        for key in state_dict.keys():
            print(f"- {key}")
        
        return state_dict

    def convert_model(self):
        if not self.selected_file or not self.ncnn_path:
            return
        
        try:
            self.status_label.setText("PyTorch → ONNX変換中...")
            self.convert_button.setEnabled(False)
            self.select_button.setEnabled(False)
            
            # Step 1: PyTorch → ONNX
            base_name = os.path.splitext(self.selected_file)[0]
            onnx_path = f"{base_name}.onnx"
            
            model = RRDBNet()
            state_dict = self.load_state_dict(self.selected_file)
            model.load_state_dict(state_dict)
            model.eval()
            
            # ダミー入力の作成
            x = torch.rand(1, 3, 64, 64)
            
            # ONNXへの変換
            with torch.no_grad():
                torch_out = torch.onnx._export(
                    model,
                    x,
                    onnx_path,
                    opset_version=11,
                    export_params=True
                )
            print(f"ONNX output shape: {torch_out.shape}")
            
            # Step 2: ONNX → NCNN
            self.status_label.setText("ONNX → NCNN変換中...")
            param_path = f"{base_name}.param"
            bin_path = f"{base_name}.bin"
            
            onnx2ncnn_path = os.path.join(self.ncnn_path, "onnx", "onnx2ncnn.exe")
            subprocess.run([
                onnx2ncnn_path,
                onnx_path,
                param_path,
                bin_path
            ], check=True)
            
            # パラメータファイルの修正
            self.status_label.setText("パラメータファイル修正中...")
            with open(param_path, 'r') as f:
                param_content = f.read()
            
            param_content = param_content.replace('input', 'data')
            
            with open(param_path, 'w') as f:
                f.write(param_content)
            
            # 中間ファイルの削除
            os.remove(onnx_path)
            
            self.status_label.setText(f"変換完了: {os.path.basename(param_path)}")
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
