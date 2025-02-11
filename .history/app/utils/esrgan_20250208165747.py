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
        
        # ベース名から全ての拡張子を除去
        name = filename
        while True:
            name, ext = os.path.splitext(name)
            if not ext:
                break
        
        # 出力ディレクトリの設定
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dirname = output_dir
        
        # 新しいファイル名の生成（出力フォーマットで拡張子を設定）
        ext = f".{output_format.lower()}" if output_format else ".png"  # デフォルトはpng
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
            
            # 出力ディレクトリの設定は呼び出し側で行う
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
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
                
                # 出力フォーマットの設定（常に明示的に指定）
                cmd.extend(["-f", output_format.lower()])
                
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
                
                # 処理完了後、出力ファイルの拡張子を確認・修正
                possible_wrong_paths = [
                    output_path + ".png",  # 例：output.jpg.png
                    output_path + ".jpg",   # 例：output.png.jpg
                    output_path + ".jpeg"   # 例：output.png.jpeg
                ]
                
                for wrong_path in possible_wrong_paths:
                    if os.path.exists(wrong_path):
                        try:
                            # 正しいパスのファイルが既に存在する場合は削除
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            os.rename(wrong_path, output_path)
                            break
                        except Exception as e:
                            self.error.emit(f"ファイル名の修正に失敗しました: {str(e)}")
            
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
                dir_name = os.path.basename(input_dir.rstrip(os.path.sep))  # 末尾のセパレータを除去
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
