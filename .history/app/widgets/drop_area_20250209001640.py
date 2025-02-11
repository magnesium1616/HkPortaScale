# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPalette, QColor

class DropArea(QWidget):
    """ドラッグ&ドロップ領域のウィジェット"""
    
    # ファイルがドロップされた時のシグナル
    files_dropped = Signal(list)  # パスのリスト
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def setup_ui(self):
        """UIの初期設定"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 説明ラベル
        self.label = QLabel(
            "ここにファイルまたはフォルダをドロップ\n"
            "またはダブルクリックして選択"
        )
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # 最小サイズの設定
        self.setMinimumHeight(200)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """ドラッグされたアイテムが領域に入った時"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # ドラッグ中のクラス設定
            self.setProperty("dragging", True)
            self.style().unpolish(self)
            self.style().polish(self)
    
    def dragLeaveEvent(self, event):
        """ドラッグされたアイテムが領域から出た時"""
        # ドラッグ中のクラス設定を解除
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def dropEvent(self, event: QDropEvent):
        """アイテムがドロップされた時"""
        urls = event.mimeData().urls()
        paths = [url.toLocalFile() for url in urls]
        self.files_dropped.emit(paths)
        
        # スタイルを元に戻す
        self.dragLeaveEvent(None)
        event.acceptProposedAction()
    
    def mouseDoubleClickEvent(self, event):
        """ダブルクリックでファイル選択ダイアログを表示"""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("画像ファイル (*.jpg *.jpeg *.png);;すべてのファイル (*.*)")
        
        if dialog.exec():
            paths = dialog.selectedFiles()
            self.files_dropped.emit(paths)
    
    def set_status(self, text: str):
        """ステータステキストの更新"""
        self.label.setText(text)
