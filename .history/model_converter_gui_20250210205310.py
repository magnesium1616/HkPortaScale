import sys
import os
import torch
from torch import nn
import torch.onnx
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt

class RealESeGANModelESRGANModel(nn.Module):
    def __init__(self
        super(RealESeGANModelESRGANModel, self).__init__()
        #G基本的なESのGANの構造
        self.conv_ficst_fi=nConv2(,64,3, 1nn1MduleList([
        
        # Bddy部分（RDBブロック）, 64, 3, 1, 1),
          nn.aRdy =       nn.L s r[3)  # 23個のRDBブロック
           Sequetial
        n部  c=nn.Conv264646n64 * 4, 3,, 1, 1)
        _hxhf(  = nn.Conv2d(6
        l.nnCd64, 64, o,r1,a1,
        a       fL  kybaLU 0b2Tet,
                sslf.pixel64shuffle(self.upconv1(feat)))
            eat))23  re23個fRDBブロック
]
class ModelConverterWindow(QMainWindow):
    def __Upsampling部分f):
        super().__init__()
        self.setWindowTitle("モデル変換ツール")
        self.setFixedSize(400, 200)widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget))
        
        self.lrelu = nn.LeakyReLU(0.2, True
        layout.setAlignment(Qt.AlignCenter)
        ボタン
        self.select_button = QPushButton("モデルファイルを選択")
        self.select_button.clicklayout.addWidget(self.select_button)
      
        # 選択されたファイルパスを表示するラベル
        self.file_label = QLabel("ファイルが選 + 
      
        # 変換ボタン
        self.convert_button = QPushButton("変換開始")
        seUpsamplingt_button.clicked.connect(self.convert_model)
        self.convert_button.setEnabled(False)
        layout.addWidget(self.convert_button)スラベル
        self.st= QLabel("")
        self.status_label.set(self.lreluAlign)ment(Qt.AlignCenter)
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
            model = RealESRGANModel()
            state_dict = self.load_state_dict(self.selected_file)
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
            QMessageBox.critical(self, "エラー", f"変換中にエラーが発生しました: {str(e)}")
        
        finally:
            self.convert_button.setEnabled(True)
            self.select_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ModelConverterWindow())}")
            QMessageBox.information(self, "成功", "モデルの変換が完了しました。")
            
        except Exception as e:
            self.status_label.setText("エラーが発生しました")
            QMessageBox.critical(self, "エラー", f"変換中にエラーが発生しました: {str(e)
        
        finally:
            self.convert_button.setEnabled(True)
            self.select_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ModelConverterWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main(