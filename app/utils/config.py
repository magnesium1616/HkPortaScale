# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Dict, Any

class Config:
    """アプリケーション設定の管理クラス"""
    
    def __init__(self, config_dir: str = "config"):
        self.base_path = self._get_base_path()
        self.config_dir = os.path.join(self.base_path, config_dir)
        self.config_file = os.path.join(self.config_dir, "settings.json")
        self.default_settings = {
            "output_suffix": "_upscaled",
            "output_dir": "upscale",
            "default_scale": 4,
            "default_format": "png",
            "jpg_quality": 80,
            "default_model": "realesrgan-x4plus-anime",
            "rename_enabled": False,
            "rename_name": "",
            "padding_digits": 4
        }
        self.current_settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """設定ファイルから設定を読み込む"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # デフォルト設定とマージ
                return {**self.default_settings, **settings}
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
        
        return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any] = None):
        """設定をファイルに保存"""
        if settings:
            self.current_settings.update(settings)
        
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"設定ファイルの保存に失敗しました: {e}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        return self.current_settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """設定値を更新して保存"""
        self.current_settings[key] = value
        self.save_settings()
    
    def update_settings(self, settings: Dict[str, Any]):
        """複数の設定を一括更新して保存"""
        self.current_settings.update(settings)
        self.save_settings()
    
    @staticmethod
    def _get_base_path() -> str:
        """アプリケーションのベースパスを取得"""
        if hasattr(sys, '_MEIPASS'):  # PyInstallerでビルドされた場合
            return sys._MEIPASS
        return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    def get_models_dir(self) -> str:
        """モデルディレクトリのパスを取得"""
        return os.path.join(self.base_path, "models")
    
    def get_executable_path(self) -> str:
        """実行ファイルのパスを取得"""
        return os.path.join(self.base_path, "realesrgan-ncnn-vulkan.exe")
