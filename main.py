from PyQt5.QtWidgets import QApplication
from filter_designer import FilterDesigner

if __name__ == "__main__":
    app = QApplication([])
    window = FilterDesigner()
    window.show()
    app.exec_()
