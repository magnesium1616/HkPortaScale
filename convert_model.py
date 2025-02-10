import torch
import sys
import os
from torch import nn
import torch.onnx

def load_state_dict(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    return state_dict

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)
    
    def forward(self, x):
        return self.conv(x)

def convert_to_onnx(model_path, output_path):
    print(f"Converting {model_path} to ONNX format...")
    
    # モデルの読み込み
    model = SimpleModel()
    state_dict = load_state_dict(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
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
    print(f"ONNX model saved to {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    
    # 出力パスの生成
    output_path = os.path.splitext(model_path)[0] + '.onnx'
    
    try:
        convert_to_onnx(model_path, output_path)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
