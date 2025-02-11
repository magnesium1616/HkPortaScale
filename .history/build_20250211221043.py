#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
from pathlib import Path
import PyInstaller.__main__

def build_exe():
    """アプリケーションのexeファイルをビルド"""
    # ビルドディレクトリの作成
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # PyInstallerコマンドの構築
    options = [
        'main.py',
        '--name=HkEzLauncher',
        '--onefile',
        '--windowed',
        '--clean',
        '--add-data=config;config',
        f'--icon={os.path.join("icon", "app_icon.ico")}',
        '--noconfirm',
    ]
    
    # ビルド実行
    PyInstaller.__main__.run(options)
    
    print("ビルドが完了しました")

if __name__ == "__main__":
    build_exe()
