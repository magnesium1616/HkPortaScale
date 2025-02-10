# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from typing import List, Optional, Tuple
from PIL import Image
from PySide6.QtCore import QObject, Signal

class ESRGANProcessor(QObject):
    """Real-ESRGAN ncnn Vulkanの実行を管理するクラス"""
    
    # シグナル定義
    progress = Signal(str)  # 進捗メッセージ
    progress_value = Signal(int)  # 進捗値（0-100）
    error = Signal(str)     # エラーメッセージ
    finished = Signal()     # 処理完了
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.process = None
        self.config = config
        self.executable = config.get_executable_path()
        self.models_dir = config.get_models_dir()
    
    def _handle_error(self, error: Exception, context: str = ""):
        """エラーハンドリング共通処理"""
        error_msg = f"エラーが発生しました"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        self.error.emit(error_msg)
    
    def _get_temp_output_path(self, output_path: str) -> str:
        """一時的な出力パスを生成"""
        return os.path.splitext(output_path)[0] + ".png"
    
    def _convert_to_jpg(self, temp_path: str, output_path: str, quality: int):
        """PNG画像をJPGに変換"""
        try:
            img = Image.open(temp_path)
            img = img.convert('RGB')  # JPEGはアルファチャンネルをサポートしないため
            img.save(output_path, "JPEG", quality=quality)
            os.remove(temp_path)  # 一時的なPNGファイルを削除
        except Exception as e:
            self.error.emit(f"JPG変換に失敗しました: {str(e)}")
            # エラーが発生した場合は、PNGファイルを最終出力として使用
            os.rename(temp_path, output_path)
    
    def generate_output_path(self, input_path: str, suffix: str, output_dir: Optional[str] = None, output_format: str = None, 
                           file_index: int = 0, rename_settings: Optional[dict] = None) -> str:
        """出力ファイルパスを生成"""
        dirname = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        
        # 出力ディレクトリの設定
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dirname = output_dir
        
        # リネーム設定の適用
        if rename_settings and rename_settings.get("rename_enabled", False):
            rename_name = rename_settings.get("rename_name", "")
            padding_digits = rename_settings.get("padding_digits", 4)
            
            if rename_name:
                # 連番を生成（例：0001）
                number = str(file_index + 1).zfill(padding_digits)
                name = f"{rename_name}{number}"
            else:
                # リネーム名が空の場合は元のファイル名を使用
                name = os.path.splitext(filename)[0]
        else:
            # リネームが無効の場合は元のファイル名を使用
            name = os.path.splitext(filename)[0]
        
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
                     output_dir: Optional[str] = None,
                     rename_settings: Optional[dict] = None) -> None:
        
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
                self.progress.emit(f"処理中 ({i}/{total_files}): {os.path.basename(input_path)}")
                
                # 出力パスの生成（リネーム設定を含む）
                output_path = self.generate_output_path(
                    input_path, suffix, output_dir, output_format,
                    i - 1, rename_settings
                )
                
                # 一時的な出力パスの生成（常にPNG）
                temp_output_path = self._get_temp_output_path(output_path)
                
                # コマンドの構築（常にPNGで出力）
                cmd = [
                    self.executable,
                    "-i", input_path,
                    "-o", temp_output_path,
                    "-n", model_name,
                    "-s", str(scale),
                    "-m", self.models_dir,
                    "-f", "png"  # 常にPNGで出力
                ]
                
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

                # 処理完了の進捗報告
                progress = int(i / total_files * 100)
                self.progress_value.emit(progress)
                
                # 出力フォーマットに応じた処理
                if output_format.lower() == "jpg":
                    self._convert_to_jpg(temp_output_path, output_path, jpg_quality)
                else:
                    # PNG出力の場合は、一時ファイルを最終的な出力パスに移動
                    os.rename(temp_output_path, output_path)
            
            # 最終進捗更新
            self.progress_value.emit(100)
            self.progress.emit("処理が完了しました")
            self.finished.emit()
            
        except Exception as e:
            self._handle_error(e)
    
    def process_directory(self,
                         input_dir: str,
                         model_name: str,
                         scale: int,
                         output_format: str = "png",
                         jpg_quality: int = 80,
                         suffix: str = "_upscaled",
                         output_dir: Optional[str] = None,
                         rename_settings: Optional[dict] = None) -> None:
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
                output_dir = os.path.join(os.path.dirname(input_dir), self.config.get_setting("output_dir"))
            
            # ファイル処理の実行
            self.process_files(
                image_files,
                model_name,
                scale,
                output_format,
                jpg_quality,
                suffix,
                output_dir,
                rename_settings
            )
            
        except Exception as e:
            self._handle_error(e, "ディレクトリの処理中")
    
    def cancel(self):
        """処理のキャンセル"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.progress.emit("処理がキャンセルされました")
