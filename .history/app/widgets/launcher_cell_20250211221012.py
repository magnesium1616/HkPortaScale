from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QMenu, QDialog, QVBoxLayout, QLineEdit, QPushButton
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QPixmap, QIcon
import os
import win32gui
import win32ui
import win32con
import win32api
from PIL import Image
import io

class LauncherCell(QWidget):
    def __init__(self, parent=None, cell_data=None):
        super().__init__(parent)
        self.parent = parent
        self.path = cell_data["path"] if cell_data else ""
        self.name = cell_data["name"] if cell_data else "ドロップしてアプリを追加"
        
        self.setup_ui()
        self.setAcceptDrops(True)
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # アイコン
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(32, 32)
        if self.path:
            self.set_icon_from_file(self.path)
        layout.addWidget(self.icon_label)
        
        # 名前
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("color: white;")
        layout.addWidget(self.name_label)
        
        layout.addStretch()
        
        # スタイル
        self.setStyleSheet("""
            QWidget {
                background-color: #3b3b3b;
                border-radius: 5px;
            }
            QWidget:hover {
                background-color: #4b4b4b;
            }
        """)
    
    def set_icon_from_file(self, path):
        try:
            # ファイルからアイコンを取得
            large, small = win32gui.ExtractIconEx(path, 0)
            if large:
                win32gui.DestroyIcon(large[0])
            if small:
                # アイコンをビットマップに変換
                hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
                hbmp = win32ui.CreateBitmap()
                hbmp.CreateCompatibleBitmap(hdc, 32, 32)
                hdc = hdc.CreateCompatibleDC()
                hdc.SelectObject(hbmp)
                hdc.DrawIcon((0, 0), small[0])
                win32gui.DestroyIcon(small[0])
                
                # ビットマップをPIL Imageに変換
                bmpstr = hbmp.GetBitmapBits(True)
                img = Image.frombuffer('RGBA', (32, 32), bmpstr, 'raw', 'BGRA', 0, 1)
                
                # PIL ImageをQPixmapに変換
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue())
                
                self.icon_label.setPixmap(pixmap)
        except:
            # アイコン取得に失敗した場合はデフォルトアイコンを使用
            self.icon_label.setText("📁")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.NoModifier and self.path:
                # 通常クリック - アプリ起動
                os.startfile(self.path)
            else:
                # ドラッグ開始
                drag = QDrag(self)
                mime_data = QMimeData()
                mime_data.setText(str(self.parent.scroll_layout.indexOf(self)))
                drag.setMimeData(mime_data)
                drag.exec_(Qt.MoveAction)
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            # ファイル/アプリのドロップ
            path = event.mimeData().urls()[0].toLocalFile()
            self.path = path
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.name_label.setText(self.name)
            self.set_icon_from_file(path)
            self.parent.settings.save_cells()
        elif event.mimeData().hasText():
            # セル位置の移動
            source_index = int(event.mimeData().text())
            target_index = self.parent.scroll_layout.indexOf(self)
            if source_index != target_index:
                widget = self.parent.scroll_layout.itemAt(source_index).widget()
                self.parent.scroll_layout.removeWidget(widget)
                self.parent.scroll_layout.insertWidget(target_index, widget)
                self.parent.settings.save_cells()
    
    def show_context_menu(self, pos):
        menu = QMenu(self)
        edit_action = menu.addAction("編集")
        delete_action = menu.addAction("削除")
        
        action = menu.exec_(self.mapToGlobal(pos))
        if action == edit_action:
            self.show_edit_dialog()
        elif action == delete_action:
            self.parent.scroll_layout.removeWidget(self)
            self.deleteLater()
            self.parent.settings.save_cells()
    
    def show_edit_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("セルの編集")
        layout = QVBoxLayout(dialog)
        
        name_edit = QLineEdit(self.name)
        layout.addWidget(QLabel("名前:"))
        layout.addWidget(name_edit)
        
        path_edit = QLineEdit(self.path)
        layout.addWidget(QLabel("パス:"))
        layout.addWidget(path_edit)
        
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("キャンセル")
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            self.name = name_edit.text()
            self.path = path_edit.text()
            self.name_label.setText(self.name)
            self.set_icon_from_file(self.path)
            self.parent.settings.save_cells() 