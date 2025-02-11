import json
import os

class Settings:
    def __init__(self):
        self.settings_file = "config/settings.json"
        self.load_settings()
    
    def load_settings(self):
        """設定ファイルを読み込む"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
            except:
                self.settings = {"cells": []}
        else:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            self.settings = {"cells": []}
            self.save_settings()
    
    def save_settings(self):
        """設定ファイルを保存する"""
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)
    
    def get_cells(self):
        """セルの設定を取得"""
        return self.settings.get("cells", [])
    
    def save_cells(self):
        """セルの設定を保存"""
        cells = []
        layout = self.parent.scroll_layout
        for i in range(layout.count()):
            cell = layout.itemAt(i).widget()
            if cell:
                cells.append({
                    "name": cell.name,
                    "path": cell.path
                })
        self.settings["cells"] = cells
        self.save_settings() 