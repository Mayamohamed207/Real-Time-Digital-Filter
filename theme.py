# gui/theme.py
from PyQt5.QtGui import QPalette, QColor

def setup_bright_theme(window):
    bright_palette = QPalette()
    bright_palette.setColor(QPalette.Window, QColor(240, 240, 240))
    bright_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    bright_palette.setColor(QPalette.Base, QColor(255, 255, 255))
    bright_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    bright_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    bright_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    bright_palette.setColor(QPalette.Text, QColor(0, 0, 0))
    bright_palette.setColor(QPalette.Button, QColor(200, 220, 255))
    bright_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    bright_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    bright_palette.setColor(QPalette.Link, QColor(0, 120, 215))
    bright_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    bright_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    window.setPalette(bright_palette)

    window.setStyleSheet('''
        QMainWindow {
            background-color: #F5F5F5;
        }
        QTabWidget::pane {
            border: 1px solid #D0D0D0;
            background: #FFFFFF;
        }
        QTabBar::tab {
            background: #E0E0E0;
            color: #000;
            padding: 8px 20px;
            margin: 2px;
            border-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #0A74DA;
            color: white;
        }
        QPushButton {
            background-color: #0A74DA;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            margin: 4px;
        }
        QPushButton:hover {
            background-color: #0C85E9;
        }
        QGroupBox {
            border: 2px solid #D0D0D0;
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 10px;
            color: #000;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QComboBox {
            background-color: #FFFFFF;
            color: #000;
            border: 1px solid #D0D0D0;
            border-radius: 4px;
            padding: 5px;
        }
        QLineEdit {
            background-color: #FFFFFF;
            color: #000;
            border: 1px solid #D0D0D0;
            border-radius: 4px;
            padding: 5px;
        }
        QTableWidget {
            background-color: #FFFFFF;
            color: #000;
            gridline-color: #D0D0D0;
            border: none;
        }
        QHeaderView::section {
            background-color: #F0F0F0;
            color: #000;
            padding: 5px;
            border: 1px solid #D0D0D0;
        }
                         QToolBar {
    
        border: none;
    }
    QToolButton {
        background-color: #00AEEF;
        color: white;
        border: 1px solid #0078D7;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 2px;
    }
    QToolButton:hover {
        background-color: #1084E0;
    }
    QToolButton:pressed {
        background-color: #005BB5;
    }
    ''')
