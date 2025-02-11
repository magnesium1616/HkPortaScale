#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
from pathlib import Path

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
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--icon=icon/app_icon.ico",
        "--name=HkPortaScale",
        "--add-data=app/resources/style.qss;app/resources",
        "--add-data=icon/app_icon.ico;icon",
        "--add-data=models;models",
        "--add-data=realesrgan-ncnn-vulkan.exe;.",
        "--add-data=vcomp140.dll;.",
        "--add-data=vcomp140d.dll;.",
        "main.py"
    ]
    
    # ビルドの実行
    subprocess.run(cmd, check=True)
    
    print("ビルドが完了しました")

if __name__ == "__main__":
    build_exe()
