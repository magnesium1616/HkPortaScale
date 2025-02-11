# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, Any

class Config:
    """アプリケーション設定の管理クラス"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "settings.json")
        self.default_settings = {
            "output_suffix": "_upscaled",
            "output_dir": "upscale",
            "default_scale": 4,
            "default_format": "png",
            "jpg_quality": 80,
            "default_model": "realesrgan-x4plus-anime"
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
