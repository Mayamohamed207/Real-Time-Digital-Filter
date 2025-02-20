from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QWidget, 
                            QButtonGroup, QRadioButton, QTabWidget, QFileDialog, QLabel, QSlider, QLineEdit, 
                            QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRectF
from PyQt5.QtGui import QPalette, QColor, QFont
import pyqtgraph as pg
import numpy as np
from scipy.signal import freqz, zpk2tf, lfilter, sosfreqz, sosfilt, sos2zpk, zpk2sos
import sys

class RealTimeTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.current_input = []
        self.current_output = []
        self.filtered_output = []
        self.signal_index = 0
        self.combined_num = np.array([1.0])  
        self.combined_den = np.array([1.0])
        self.filter_num = [1.0]
        self.filter_den = [1.0]
        self.filter_state = None
        self.processing_speed = 10  # Processing speed in points per second
        self.init_ui()

    def toggle_input_method(self):
        if self.file_input_radioButton.isChecked():
            self.mouse_pad.setEnabled(False)
            # Only load file if radio button was just checked
            if self.file_input_radioButton.isChecked():
                self.load_signal()
        else:
            self.mouse_pad.setEnabled(True)
            self.clear_graphs()
            # Start processing automatically for touch pad
            if not self.processing_timer.isActive():
                self.processing_timer.start(1000 // self.processing_speed)

    def load_signal(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Signal", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if filename:
            try:
                data = np.loadtxt(filename, skiprows=1)  # Skip header row if it exists
                self.current_input = data.tolist()
                self.current_output = []
                self.filtered_output = []
                self.signal_index = 0
                self.filter_state = None
                # Start processing if using file input
                if self.file_input_radioButton.isChecked():
                    self.processing_timer.start(1000 // self.processing_speed)
            except Exception as e:
                print(f"Error loading signal: {e}")
                
    def clear_graphs(self):
        self.current_input = []
        self.current_output = []
        self.filtered_output = []
        self.signal_index = 0
        self.filter_state = None
        # Clear all plots
        self.input_plot.clear()
        self.filtered_plot.clear()

    def on_mouse_move(self, event):
        if self.touch_pad_radioButton.isChecked():
            pos = self.mouse_pad.getViewBox().mapSceneToView(event)
            self.current_input.append(pos.y())

    def update_filter_from_design(self):
        design_tab = self.parent.findChild(QWidget, "design_tab")
        if design_tab:
            # Get zeros and poles from design tab
            self.combined_num, self.combined_den = zpk2tf(design_tab.zeros, design_tab.poles, 1.0)
            self.filter_state = None
            
            # Also include all-pass filters if they're enabled
            for i, allpass in enumerate(design_tab.allpass_filters):
                checkbox = design_tab.allpass_table.cellWidget(i, 1)
                if checkbox and checkbox.isChecked():
                    # Get allpass coefficients
                    a = allpass['a']
                    zero = 1/np.conj(a)
                    # Convert allpass to transfer function
                    num_allpass = [zero]
                    den_allpass = [a]
                    b_allpass, a_allpass = zpk2tf([zero], [a], 1.0)
                    
                    # Convolve with existing filter
                    self.combined_num = np.convolve(self.combined_num, b_allpass)
                    self.combined_den = np.convolve(self.combined_den, a_allpass)

    def process_next_point(self):
        if not self.current_input or self.signal_index >= len(self.current_input):
            return

        # Ensure filter coefficients are updated
        self.update_filter_from_design()

        # Process the next input point
        input_point = self.current_input[self.signal_index]

        # Apply the filter
        if self.filter_state is None:
            self.filter_state = np.zeros(max(len(self.combined_num), len(self.combined_den)) - 1)

        output_point, self.filter_state = lfilter(
            self.combined_num, self.combined_den,
            [input_point], zi=self.filter_state
        )

        # Append the real part of the filtered output
        self.current_output.append(float(input_point))
        self.filtered_output.append(float(output_point[0].real))  # Use real part only

        # Plot the results
        display_window = 500  # Number of points to display in the plot
        start_idx = max(0, self.signal_index - display_window)

        self.input_plot.clear()
        self.filtered_plot.clear()
        self.input_plot.plot(range(start_idx, self.signal_index + 1),
                            self.current_input[start_idx:self.signal_index + 1],
                            pen=pg.mkPen('r'))  # Red for input signal
        self.filtered_plot.plot(range(start_idx, self.signal_index + 1),
                                self.filtered_output[start_idx:self.signal_index + 1],
                                pen=pg.mkPen('b'))  # Blue for filtered signal

        self.signal_index += 1



    def update_speed(self):
        self.processing_speed = self.speed_slider.value()
        if self.processing_timer.isActive():
            self.processing_timer.setInterval(1000 // self.processing_speed)

    def on_checkbox_toggled(self, button, checked):
        if button == self.add_zero_checkbox and checked:
            self.add_pole_checkbox.setChecked(False)
        elif button == self.add_pole_checkbox and checked:
            self.add_zero_checkbox.setChecked(False)

    def update_filter(self, num, den):
        self.filter_num = num
        self.filter_den = den
        self.filter_state = None  # Reset filter state
        
        self.filtered_output = []
        self.signal_index = 0
        
        # Reprocess all existing input with new filter
        if self.current_input:
            self.filter_state = np.zeros(max(len(self.filter_num), len(self.filter_den)) - 1)
            filtered_signal, self.filter_state = lfilter(
                self.filter_num, self.filter_den,
                self.current_input, zi=self.filter_state
            )
            self.filtered_output = filtered_signal.tolist()
            self.signal_index = len(self.current_input)










    def init_ui(self):
        layout = QVBoxLayout()

        input_group = QHBoxLayout()
        self.touch_pad_radioButton = QRadioButton("Touch Pad")
        self.file_input_radioButton = QRadioButton("Load Signal")
        input_group.addWidget(self.touch_pad_radioButton)
        input_group.addWidget(self.file_input_radioButton)
        self.touch_pad_radioButton.setChecked(True)
        
        # Connect radio buttons
        self.touch_pad_radioButton.toggled.connect(self.toggle_input_method)
        self.file_input_radioButton.toggled.connect(self.toggle_input_method)
        
        # Mouse pad
        self.mouse_pad = pg.PlotWidget(title="Touch Pad Input Area")
        self.mouse_pad.setFixedHeight(200)
        self.mouse_pad.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.mouse_pad.setBackground('w')
        self.mouse_pad.getPlotItem().getAxis('bottom').setVisible(False)  # Removes the x-axis
        self.mouse_pad.getPlotItem().getAxis('left').setVisible(False)
   
        
        # Signal plots
        self.input_plot = pg.PlotWidget(title="Input Signal")
        self.input_plot.setBackground('w')
        self.input_plot.showGrid(x=True, y=True)
        self.filtered_plot = pg.PlotWidget(title="Filtered Signal")
        self.filtered_plot.setBackground('w')
        self.filtered_plot.showGrid(x=True, y=True)
        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Processing Speed (points/sec):")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(self.processing_speed)
        self.speed_slider.setStyleSheet('''QSlider::handle:horizontal { background: #0A74DA; width: 20px; height: 20px; border-radius: 10px; }''')  # Custom slider style
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        
        # Add components to layout
        layout.addLayout(input_group)
        layout.addWidget(self.mouse_pad)
        layout.addWidget(self.input_plot)
        layout.addWidget(self.filtered_plot)
        layout.addLayout(speed_layout)
        
        self.setLayout(layout)
        
        # Initialize processing timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_next_point)
        self.processing_timer.start(1000 // self.processing_speed)

