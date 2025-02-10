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
    progress_value = Signal(int)  # 進捗値（0-100）
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
    
    def generate_output_path(self, input_path: str, suffix: str, output_dir: Optional[str] = None, output_format: str = None) -> str:
        """出力ファイルパスを生成"""
        dirname = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # 出力ディレクトリの設定
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dirname = output_dir
        
        # 新しいファイル名の生成（出力フォーマットが指定されている場合は拡張子を変更）
        if output_format:
            ext = f".{output_format.lower()}"
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
            
            # 複数ファイルの場合、最初のファイル名をベースにフォルダを作成
            if total_files > 1 and not output_dir:
                first_file_dir = os.path.dirname(input_paths[0])
                first_file_name = os.path.splitext(os.path.basename(input_paths[0]))[0]
                output_dir = os.path.join(first_file_dir, f"{first_file_name}_upscaled")
            
            for i, input_path in enumerate(input_paths, 1):
                # 進捗報告
                progress = int((i - 1) / total_files * 100)
                self.progress_value.emit(progress)
                self.progress.emit(f"処理中: {os.path.basename(input_path)}")
                
                # 出力パスの生成
                output_path = self.generate_output_path(input_path, suffix, output_dir, output_format)
                
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
                else:
                    cmd.extend(["-f", "png"])  # 明示的にPNGを指定
                
                # プロセスの実行（ウィンドウを非表示に）
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    startupinfo=startupinfo
                )
                
                # 処理の完了を待機
                stdout, stderr = self.process.communicate()
                
                # エラーチェック
                if self.process.returncode != 0:
                    error_msg = f"エラー: {os.path.basename(input_path)}"
                    self.error.emit(error_msg)
                    continue
            
            # 最終進捗更新
            self.progress_value.emit(100)
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
            
            # 出力ディレクトリの設定（ディレクトリ名をベースに作成）
            if not output_dir:
                dir_name = os.path.basename(input_dir)
                output_dir = os.path.join(os.path.dirname(input_dir), f"{dir_name}_upscaled")
            
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
