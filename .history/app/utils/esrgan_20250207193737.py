# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from typing import List, Optional, Tuple
from PySide6.QtCore import QObject, Signal

class ESRGANProcessor(QObject):
    """Real-ESRGAN ncnn Vulkanの実行を管理するクラス"""
    
    # シグナル定義
    progress = Signal(str)  # 進捗メッセージ
    error = Signal(str)     # エラーメッセージ
    finished = Signal()     # 処理完了
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        
        # 実行ファイルのパスを取得
        if hasattr(sys, '_MEIPASS'):  # PyInstallerでビルドされた場合
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        self.executable = os.path.join(base_path, "realesrgan-ncnn-vulkan.exe")
        self.models_dir = os.path.join(base_path, "models")
    
    def generate_output_path(self, input_path: str, suffix: str, output_dir: Optional[str] = None) -> str:
        """出力ファイルパスを生成"""
        dirname = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # 出力ディレクトリの設定
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dirname = output_dir
        
        # 新しいファイル名の生成
        new_name = f"{name}{suffix}{ext}"
        return os.path.join(dirname, new_name)
    
    def process_files(self, 
                     input_paths: List[str],
                     model_name: str,
                     scale: int,
                     output_format: str = "png",
                     jpg_quality: int = 80,
                     suffix: str = "_upscaled",
                     output_dir: Optional[str] = None) -> None:
        """ファイルのアップスケール処理"""
        try:
            total_files = len(input_paths)
            for i, input_path in enumerate(input_paths, 1):
                # 進捗報告
                self.progress.emit(f"処理中 ({i}/{total_files}): {os.path.basename(input_path)}")
                
                # 出力パスの生成
                output_path = self.generate_output_path(input_path, suffix, output_dir)
                
                # モデルファイルのパスを構築
                model_path = os.path.join(self.models_dir, model_name)
                
                # コマンドの構築
                cmd = [
                    self.executable,
                    "-i", input_path,
                    "-o", output_path,
                    "-n", model_name,
                    "-s", str(scale),
                    "-m", self.models_dir
                ]
                
                # 出力フォーマットの設定
                if output_format.lower() == "jpg":
                    cmd.extend(["-f", "jpg"])
                    # JPEGの品質設定（現在のバージョンでは未サポート）
                    # 将来的にサポートされる可能性があるため、コメントとして残す
                    # cmd.extend(["-q", str(jpg_quality)])
                
                # プロセスの実行
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 処理の完了を待機
                stdout, stderr = self.process.communicate()
                
                # エラーチェック
                if self.process.returncode != 0:
                    error_msg = (
                        f"エラー ({os.path.basename(input_path)}):\n"
                        f"コマンド: {' '.join(cmd)}\n"
                        f"終了コード: {self.process.returncode}\n"
                        f"標準出力: {stdout}\n"
                        f"エラー出力: {stderr}"
                    )
                    self.error.emit(error_msg)
                    continue
            
            self.progress.emit("処理が完了しました")
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"エラーが発生しました: {str(e)}")
    
    def process_directory(self,
                         input_dir: str,
                         model_name: str,
                         scale: int,
                         output_format: str = "png",
                         jpg_quality: int = 80,
                         suffix: str = "_upscaled",
                         output_dir: Optional[str] = None) -> None:
        """ディレクトリ内の画像ファイルを処理"""
        try:
            # 画像ファイルの収集
            image_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
            
            # 出力ディレクトリの設定
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(input_dir), "upscale")
            
            # ファイル処理の実行
            self.process_files(
                image_files,
                model_name,
                scale,
                output_format,
                jpg_quality,
                suffix,
                output_dir
            )
            
        except Exception as e:
            self.error.emit(f"ディレクトリの処理中にエラーが発生しました: {str(e)}")
    
    def cancel(self):
        """処理のキャンセル"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.progress.emit("処理がキャンセルされました")
