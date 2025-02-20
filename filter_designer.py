from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtGui import QPalette, QColor
from design_tab import DesignTab
from real_time_tab import RealTimeTab
from theme import setup_bright_theme
import numpy as np
from scipy.signal import freqz, zpk2tf, lfilter, sosfreqz, sosfilt, sos2zpk, zpk2sos
import sys

class FilterDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Filter Designer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.zeros = []
        self.poles = []
        self.allpass_filters = []
        self.combined_num = [1.0]
        self.combined_den = [1.0]
        self.current_filter_num = [1.0]  
        self.current_filter_den = [1.0]
        
        setup_bright_theme(self)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        self.design_tab = DesignTab(self)
        self.design_tab.setObjectName("design_tab") 
        self.tabs.addTab(self.design_tab, "Design Tab")
        self.real_time_tab = RealTimeTab(self)
        
        self.tabs.addTab(self.design_tab, "Filter Design")
        self.tabs.addTab(self.real_time_tab, "Real-Time Processing")
        
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)

    def update_filter_coefficients(self, num, den):
        self.current_filter_num = num
        self.current_filter_den = den
        self.real_time_tab.update_filter(num, den)



