from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QCheckBox, QPushButton, QToolBar,QWidget, QButtonGroup, QRadioButton, QTabWidget, 
    QFileDialog, QLabel, QSlider, QLineEdit, QTableWidget, QTableWidgetItem, 
    QGroupBox, QComboBox, QFrame, QAction,QScrollArea, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QCheckBox, QPushButton, QWidget, QButtonGroup, QRadioButton, QTabWidget, 
    QFileDialog, QLabel, QSlider, QLineEdit, QTableWidget, QTableWidgetItem, 
    QGroupBox, QComboBox, QFrame, QScrollArea, QSplitter)

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPalette, QColor, QFont
import pyqtgraph as pg
import numpy as np
from scipy.signal import freqz, zpk2tf, lfilter, sosfreqz, sosfilt, sos2zpk, zpk2sos
import sys
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages






class DesignTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.history = []
        self.future = []
        self.zeros = []
        self.poles = []
        self.filter_library = {
            "Butterworth LPF": {"order": 4, "type": "butter", "btype": "low"},
            "Butterworth HPF": {"order": 4, "type": "butter", "btype": "high"},
            "Butterworth BPF": {"order": 4, "type": "butter", "btype": "band"},
            "Chebyshev-1 LPF": {"order": 4, "type": "cheby1", "rp": 1, "btype": "low"},
            "Chebyshev-1 HPF": {"order": 4, "type": "cheby1", "rp": 1, "btype": "high"},
            "Chebyshev-1 BPF": {"order": 4, "type": "cheby1", "rp": 1, "btype": "band"},
            "Chebyshev-2 LPF": {"order": 4, "type": "cheby2", "rs": 40, "btype": "low"},
            "Chebyshev-2 HPF": {"order": 4, "type": "cheby2", "rs": 40, "btype": "high"},
            "Elliptic LPF": {"order": 4, "type": "ellip", "rp": 1, "rs": 40, "btype": "low"},
            "Elliptic HPF": {"order": 4, "type": "ellip", "rp": 1, "rs": 40, "btype": "high"}
        }
        self.dragging_item = None
        self.dragging_type = None
        self.dragging_index = -1
        self.add_conjugates = False
        self.allpass_filters = []
        
        self.allpass_library = {
            "Mild Phase Shift (a=0.5)": 0.5,
            "Strong Phase Shift (a=0.9)": 0.9,
            "Negative Phase Shift (a=-0.5)": -0.5,
            "Complex Phase Shift (a=0.5+0.5j)": 0.5 + 0.5j,
        }
        
        self.init_ui()

    def load_library_filter(self):
        selected_filter = self.filter_library[self.filter_combo.currentText()]
        
        # Get filter parameters
        filter_type = selected_filter["type"]
        order = selected_filter["order"]
        btype = selected_filter["btype"]
        
        # Set default frequency parameters
        wp = 0.25  # passband frequency (normalized)
        ws = 0.35  # stopband frequency (normalized)
        if btype == "band":
            wp = [0.2, 0.3]
            ws = [0.1, 0.4]
        
        # Create the filter based on type
        if filter_type == "butter":
            z, p, k = signal.butter(order, wp, btype=btype, output='zpk')
        elif filter_type == "cheby1":
            rp = selected_filter["rp"]
            z, p, k = signal.cheby1(order, rp, wp, btype=btype, output='zpk')
        elif filter_type == "cheby2":
            rs = selected_filter["rs"]
            z, p, k = signal.cheby2(order, rs, ws, btype=btype, output='zpk')
        elif filter_type == "ellip":
            rp = selected_filter["rp"]
            rs = selected_filter["rs"]
            z, p, k = signal.ellip(order, rp, rs, wp, btype=btype, output='zpk')
        
        # Update zeros and poles
        self._save_state()
        self.zeros = list(z)
        self.poles = list(p)
        self.plot_z_plane()

    def setup_unit_circle(self):
        theta = np.linspace(0, 2 * np.pi, 500)
        unit_circle_x = np.cos(theta)
        unit_circle_y = np.sin(theta)
        unit_circle_pen = pg.mkPen(color='orange', width=2)
        self.unit_circle=self.z_plane_plot.plot(unit_circle_x, unit_circle_y, pen=unit_circle_pen)
        
        


    def on_checkbox_toggled(self, button, checked):
        if button == self.add_zero_checkbox and checked:
            self.add_pole_checkbox.setChecked(False)
        elif button == self.add_pole_checkbox and checked:
            self.add_zero_checkbox.setChecked(False)
    
    def load_filter(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Filter", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'r') as f:
                lines = f.readlines()[1:] 
                self.zeros = []
                self.poles = []
                for line in lines:
                    zero, pole = line.strip().split(',')
                    self.zeros.append(complex(zero))
                    self.poles.append(complex(pole))

            self.plot_z_plane()
            
    def toggle_conjugates(self, checked):
        self.add_conjugates = checked

    def preview_library_filter(self):
        a = self.allpass_library[self.allpass_combo.currentText()]
        dialog = AllPassPreviewDialog(a, self)
        dialog.exec_()
    
    def preview_allpass_filter(self):
        try:
            # Get coefficient from either library or custom input
            if self.custom_a_input.text():
                a_text = self.custom_a_input.text()
                if 'j' in a_text:
                    a = complex(a_text)
                else:
                    a = float(a_text)
            else:
                a = self.allpass_library[self.allpass_combo.currentText()]
            
            dialog = AllPassPreviewDialog(a, self)
            dialog.exec_()
        except ValueError:
            pass
    
    def add_allpass_filter(self):
        try:
            # Get coefficient from either library or custom input
            if self.custom_a_input.text():
                a_text = self.custom_a_input.text()
                if 'j' in a_text:
                    a = complex(a_text)
                else:
                    a = float(a_text)
            else:
                a = self.allpass_library[self.allpass_combo.currentText()]
            
            # Add to table
            row = self.allpass_table.rowCount()
            self.allpass_table.insertRow(row)
            
            # Add coefficient
            self.allpass_table.setItem(row, 0, QTableWidgetItem(str(a)))
         
            enable_checkbox = QCheckBox()
            enable_checkbox.setChecked(True)
            enable_checkbox.stateChanged.connect(self.update_response)
            self.allpass_table.setCellWidget(row, 1, enable_checkbox)
            
     
            preview_btn = QPushButton("Preview")
            preview_btn.clicked.connect(lambda: AllPassPreviewDialog(a, self).exec_())
            self.allpass_table.setCellWidget(row, 2, preview_btn)
            
          
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda: self.remove_allpass(row))
            self.allpass_table.setCellWidget(row, 3, remove_btn)
            
            # Store filter data
            self.allpass_filters.append({
                'a': a,
                'zero': 1/np.conj(a),
                'pole': a
            })
               # Add to zeros and poles if enabled
            if enable_checkbox.isChecked():
                self.zeros.append(1/np.conj(a))  # Add zero
                self.poles.append(a)  # Add pole
                self.plot_z_plane()  # Update
            self.update_response()
            
        except ValueError:
           pass
    
    def remove_allpass(self, row):
      
        removed_filter = self.allpass_filters[row]
        checkbox = self.allpass_table.cellWidget(row, 1)
        
        # If the filter was enabled, remove its zero and pole from the z-plane
        if checkbox and checkbox.isChecked():
            try:
                self.zeros.remove(removed_filter['zero'])
                self.poles.remove(removed_filter['pole'])
            except ValueError:
                pass  # In case the zero/pole was already removed
        
        # Remove from table and filters list
        self.allpass_table.removeRow(row)
        self.allpass_filters.pop(row)
        
        # Update plots
        self.plot_z_plane()
        self.update_response()
    
    def update_response(self):
        # Combine main filter with enabled all-pass filters
        all_zeros = self.zeros.copy()
        all_poles = self.poles.copy()
        
        # Add zeros and poles from enabled all-pass filters
        for i, allpass in enumerate(self.allpass_filters):
            checkbox = self.allpass_table.cellWidget(i, 1)
            if checkbox and checkbox.isChecked():
                all_zeros.append(allpass['zero'])
                all_poles.append(allpass['pole'])
        
        # Calculate combined response
        if all_zeros or all_poles:
            w, h = freqz(zpk2tf(all_zeros, all_poles, 1)[0],
                        zpk2tf(all_zeros, all_poles, 1)[1])
            
            magnitude = 20 * np.log10(np.abs(h))
            phase = np.unwrap(np.angle(h))
            
            self.magnitude_plot.clear()
            self.phase_plot.clear()
            self.magnitude_plot.plot(w, magnitude, pen=pg.mkPen('b', width=2))
            self.phase_plot.plot(w, phase, pen=pg.mkPen('r', width=2))

    def clear_zeros(self):
        self._save_state()
        self.zeros.clear()
        self.plot_z_plane()

    def clear_poles(self):
        self._save_state()
        self.poles.clear()
        self.plot_z_plane()

    def swap_zeros_poles(self):
        self._save_state()
        self.zeros, self.poles = self.poles, self.zeros
        self.plot_z_plane()

    def undo(self):
        if self.history:
            self.future.append((
                self.zeros.copy(),
                self.poles.copy(),
                [filter.copy() for filter in self.allpass_filters]
            ))
            self.zeros, self.poles, self.allpass_filters = self.history.pop()
            self.refresh_allpass_table()
            self.plot_z_plane()

    def redo(self):
        if self.future:
            self.history.append((
                self.zeros.copy(),
                self.poles.copy(),
                [filter.copy() for filter in self.allpass_filters]
            ))
            self.zeros, self.poles, self.allpass_filters = self.future.pop()
            self.refresh_allpass_table()
            self.plot_z_plane()


    def _save_state(self):
        self.history.append((
            self.zeros.copy(),
            self.poles.copy(),
            [filter.copy() for filter in self.allpass_filters]  # Deep copy of filters
        ))
        self.future.clear()

    def export_filter_diagrams(self):
        
        if not self.zeros and not self.poles:
            print("No filter design to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export Filter Diagrams", "", "PDF Files (*.pdf)")
        if not filename:
            return
        
        print(f"Filename chosen: {filename}")

        # Compute coefficients
        b, a = zpk2tf(self.zeros, self.poles, 1)  # Direct Form II coefficients
        sos = zpk2sos(self.zeros, self.poles, 1)  # Cascade Form sections

        try:
            with PdfPages(filename) as pdf:
                # Direct Form II
                fig, ax = plt.subplots(figsize=(10, 6))
                self.draw_direct_form_ii(ax, b, a)
                pdf.savefig(fig)
                plt.close(fig)

                # Cascade Form
                fig, ax = plt.subplots(figsize=(10, 6))
                self.draw_cascade_form(ax, sos)
                pdf.savefig(fig)
                plt.close(fig)

            print(f"Filter diagrams successfully exported to: {filename}")
        except Exception as e:
            print(f"Failed to export filter diagrams: {e}")


    def draw_direct_form_ii(self, ax, b, a):
       
        ax.set_title("Direct Form II Block Diagram", fontsize=16)
        ax.axis("off")

        ax.text(-0.1, 0.7, r"$x[n]$", fontsize=14, ha="center")
        ax.arrow(0.0, 0.7, 0.1, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")

        # Output y[n]
        ax.text(1.4, 0.7, r"$y[n]$", fontsize=14, ha="center")

        # Summation points
        ax.plot(0.2, 0.7, "o", color="black")  # Input summation point
        ax.plot(0.8, 0.7, "o", color="black")  # Output summation point

        # Delays
        ax.text(0.4, 0.5, r"$Z^{-1}$", fontsize=12, ha="center", bbox=dict(boxstyle="circle", edgecolor="black"))
        ax.text(0.6, 0.3, r"$Z^{-1}$", fontsize=12, ha="center", bbox=dict(boxstyle="circle", edgecolor="black"))

        # Feedforward coefficients
        ax.text(0.4, 0.8, f"$b_0={b[0]:.2f}$", fontsize=12)
        ax.text(0.6, 0.8, f"$b_1={b[1]:.2f}$", fontsize=12)
        ax.text(0.8, 0.8, f"$b_2={b[2]:.2f}$", fontsize=12)

        # Feedback coefficients
        ax.text(0.4, 0.4, f"$-a_1={-a[1]:.2f}$", fontsize=12)
        ax.text(0.6, 0.2, f"$-a_2={-a[2]:.2f}$", fontsize=12)

        # Arrows and connections
        ax.arrow(0.2, 0.7, 0.15, -0.2, head_width=0.03, head_length=0.03, fc="black", ec="black")  # First delay
        ax.arrow(0.4, 0.5, 0.15, -0.2, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Second delay
        ax.arrow(0.4, 0.7, 0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Feedforward b1
        ax.arrow(0.6, 0.7, 0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Feedforward b2
        ax.arrow(0.4, 0.5, -0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Feedback -a1
        ax.arrow(0.6, 0.3, -0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Feedback -a2

        # Final connections
        ax.arrow(0.8, 0.7, 0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Output arrow
        ax.arrow(0.2, 0.7, 0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # First connection to b0
        ax.arrow(0.4, 0.7, 0.2, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")  # Connection to b1



    def draw_cascade_form(self,ax, sos):
        
        ax.set_title("Cascade Form Block Diagram", fontsize=16)
        ax.axis('off')

        num_sections = len(sos)  # Number of biquads
        x_spacing = 1.5  # Horizontal spacing between sections
        y_mid = 0.5  # Vertical midpoint for the diagram

        for i, section in enumerate(sos):
            b = section[:3]  # Feedforward coefficients
            a = section[3:]  # Feedback coefficients

            # Position for the current section
            x_offset = i * x_spacing

            # Draw feedforward coefficients
            ax.text(x_offset + 0.2, y_mid + 0.1, f"$b_0={b[0]:.2f}$", fontsize=10)
            ax.text(x_offset + 0.2, y_mid + 0.2, f"$b_1={b[1]:.2f}$", fontsize=10)
            ax.text(x_offset + 0.2, y_mid + 0.3, f"$b_2={b[2]:.2f}$", fontsize=10)

            # Draw feedback coefficients
            ax.text(x_offset + 0.2, y_mid - 0.2, f"$a_1={a[1]:.2f}$", fontsize=10)
            ax.text(x_offset + 0.2, y_mid - 0.3, f"$a_2={a[2]:.2f}$", fontsize=10)

            # Draw section box
            ax.add_patch(
                plt.Rectangle(
                    (x_offset, y_mid - 0.15),  # Bottom-left corner
                    1.0,  # Width
                    0.3,  # Height
                    edgecolor="black",
                    facecolor="lightblue",
                    lw=1.5,
                )
            )
            ax.text(x_offset + 0.5, y_mid, f"Biquad {i + 1}", fontsize=10, ha="center", va="center")

            # Draw arrows to connect sections
            if i < num_sections - 1:
                ax.arrow(
                    x_offset + 1.0, y_mid,  # Start point
                    0.5, 0,  # Arrow dimensions
                    head_width=0.05, head_length=0.1, fc="black", ec="black"
                )

        # Add input and output labels
        ax.text(-0.5, y_mid, r"$x[n]$", fontsize=14, ha="center", va="center")
        ax.arrow(-0.3, y_mid, 0.3, 0, head_width=0.05, head_length=0.1, fc="black", ec="black")
        ax.text(num_sections * x_spacing + 0.2, y_mid, r"$y[n]$", fontsize=14, ha="center", va="center")
    
    def refresh_allpass_table(self):
        # Clear the table
        self.allpass_table.setRowCount(0)
        
 
        for allpass in self.allpass_filters:
            row = self.allpass_table.rowCount()
            self.allpass_table.insertRow(row)
            
            # Add coefficient
            self.allpass_table.setItem(row, 0, QTableWidgetItem(str(allpass['a'])))
    
            enable_checkbox = QCheckBox()
            enable_checkbox.setChecked(True)  
            enable_checkbox.stateChanged.connect(self.update_response)
            self.allpass_table.setCellWidget(row, 1, enable_checkbox)
            
  
            preview_btn = QPushButton("Preview")
            preview_btn.clicked.connect(lambda: AllPassPreviewDialog(allpass['a'], self).exec_())
            self.allpass_table.setCellWidget(row, 2, preview_btn)

            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda: self.remove_allpass(row))
            self.allpass_table.setCellWidget(row, 3, remove_btn)


    def plot_z_plane(self):
        self.z_plane_plot.clear()
        self.z_plane_plot.addItem(self.unit_circle)

        for zero in self.zeros:
            self.z_plane_plot.plot([zero.real], [zero.imag], pen=None, symbol="o", symbolBrush="b", symbolSize=12)  # Increase size

        for pole in self.poles:
            self.z_plane_plot.plot([pole.real], [pole.imag], pen=None, symbol="x", symbolBrush="r", symbolSize=12)  # Increase size


        self.plot_frequency_response()
       
      

    
    def plot_frequency_response(self):
        if not self.zeros and not self.poles:
            self.magnitude_plot.clear()
            self.phase_plot.clear()
            return

        z, p = self.zeros, self.poles
        numerator, denominator = zpk2tf(z, p, 1)
        w, h = freqz(numerator, denominator)
        magnitude = 20 * np.log10(abs(h))
        phase = np.angle(h)

        self.magnitude_plot.clear()
        self.phase_plot.clear()
        magnitude_line_pen = pg.mkPen(color='blue', width=2) 
        self.magnitude_plot.plot(w, magnitude, pen=magnitude_line_pen)
        phase_line_pen = pg.mkPen(color='red', width=2)
        self.phase_plot.plot(w, phase, pen=phase_line_pen)

    def add_allpass(self):
        row_position = self.allpass_list.rowCount()
        self.allpass_list.insertRow(row_position)
        coeff_input = QLineEdit("0.0")
        enable_checkbox = QCheckBox()
        self.allpass_list.setCellWidget(row_position, 0, coeff_input)
        self.allpass_list.setCellWidget(row_position, 1, enable_checkbox)

    def save_filter(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Filter", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w') as f:
                f.write("Zeros,Poles\n")
                for zero, pole in zip(self.zeros, self.poles):
                    f.write(f"{zero},{pole}\n")


    def generate_c_code(self):
        
        if not self.zeros and not self.poles:
            print("No filter design available to generate C code.")
            return

        # Generate C code
        c_code = self.create_c_code()

        # Open a save dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save C Code", "", "C Files (*.c);;All Files (*)", options=options
        )

        if file_path:
            with open(file_path, "w") as file:
                file.write(c_code)
            print(f"C code saved to {file_path}")

    def create_c_code(self):
        
        z, p = self.zeros, self.poles
        numerator, denominator = zpk2tf(z, p, 1.0)

        c_code = f"""

    #include <stdio.h>

    #define FILTER_ORDER {len(numerator) - 1}

    float filter_coeffs_num[FILTER_ORDER + 1] = {{ {', '.join(f'{coef:.6f}' for coef in numerator)} }};
    float filter_coeffs_den[FILTER_ORDER + 1] = {{ {', '.join(f'{coef:.6f}' for coef in denominator)} }};

    float apply_filter(float input, float *state) {{
        float output = 0.0;

        // Apply the numerator coefficients
        for (int i = 0; i <= FILTER_ORDER; i++) {{
            output += filter_coeffs_num[i] * (i == 0 ? input : state[i - 1]);
        }}

        // Apply the denominator coefficients
        for (int i = 1; i <= FILTER_ORDER; i++) {{
            output -= filter_coeffs_den[i] * state[i - 1];
        }}

        // Update state
        for (int i = FILTER_ORDER - 1; i > 0; i--) {{
            state[i] = state[i - 1];
        }}
        state[0] = input;

        return output;
    }}

    int main() {{
        float state[FILTER_ORDER] = {{0}};
        float input_sample = 1.0; // Example input
        float output_sample = apply_filter(input_sample, state);

        printf("Output Sample: %f\\n", output_sample);
        return 0;
    }}
    """
        return c_code

    
    def on_mouse_click(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            vb = self.z_plane_plot.getViewBox()
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            clicked_type, clicked_index = self.find_clicked_point(x, y)
            if clicked_type:
                self.dragging_item = complex(x, y)
                self.dragging_type = clicked_type
                self.dragging_index = clicked_index
                self.z_plane_plot.setMouseEnabled(False, False)
            else: 
                new_point = complex(x, y)
                if self.add_zero_checkbox.isChecked():
                    self.zeros.append(new_point)
                    if self.add_conjugates and abs(y) > 0.001:
                        self.zeros.append(complex(x, -y))
                elif self.add_pole_checkbox.isChecked():
                    self.poles.append(new_point)
                    if self.add_conjugates and abs(y) > 0.001:
                        self.poles.append(complex(x, -y))
                self.plot_z_plane()

    def mouseMoveEvent(self, event):
        if self.dragging_item is not None:
            pos = self.z_plane_plot.mapToScene(event.pos())
            vb = self.z_plane_plot.getViewBox()
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            if self.dragging_type == 'zero':
                self.zeros[self.dragging_index] = complex(x, y)
            elif self.dragging_type == 'pole':
                self.poles[self.dragging_index] = complex(x, y)

            self.plot_z_plane()

    def mouseReleaseEvent(self, event):
        if self.dragging_item is not None:
            self.dragging_item = None
            self.dragging_type = None
            self.dragging_index = -1
            self.z_plane_plot.setMouseEnabled(True, True)

    def find_clicked_point(self, x, y, threshold=0.05):
        # Check zeros
        for i, zero in enumerate(self.zeros):
            if abs(zero.real - x) < threshold and abs(zero.imag - y) < threshold:
                return 'zero', i

        # Check poles
        for i, pole in enumerate(self.poles):
            if abs(pole.real - x) < threshold and abs(pole.imag - y) < threshold:
                return 'pole', i

        return None, -1
    
    def toggle_allpass_filter(self, state):
        if state == Qt.Checked:  
            for filter_data in self.allpass_filters:
                self.zeros.append(filter_data['zero'])
                self.poles.append(filter_data['pole'])
        else: 
            for filter_data in self.allpass_filters:
                if filter_data['zero'] in self.zeros:
                    self.zeros.remove(filter_data['zero'])
                if filter_data['pole'] in self.poles:
                    self.poles.remove(filter_data['pole'])
        self.plot_z_plane()  
        self.update_response()

    def init_ui(self):
        main_layout = QVBoxLayout()

        top_bar = QToolBar("Main Toolbar")
        top_bar.setMovable(False)
        buttons = [
            ("Clear Zeros", self.clear_zeros),
            ("Clear Poles", self.clear_poles),
            ("Swap Zeros/Poles", self.swap_zeros_poles),
            ("Undo", self.undo),
            ("Redo", self.redo),
            ("Save", self.save_filter),
            ("Load", self.load_filter),
            ("Generate C Code", self.generate_c_code),
        ]
        for text, callback in buttons:
            btn_action = QAction(text, self)
            btn_action.triggered.connect(callback)
            top_bar.addAction(btn_action)
        main_layout.addWidget(top_bar)

        main_splitter = QSplitter(Qt.Horizontal)

        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(250)

        center_panel = self.create_center_panel()

        right_panel = self.create_right_panel()

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([250, 400, 400])

        main_layout.addWidget(main_splitter)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.setLayout(main_layout)


    def create_left_panel(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.add_zero_checkbox = QCheckBox("Zero")
        self.add_pole_checkbox = QCheckBox("Pole")
        self.add_zero_checkbox.setChecked(True)
        
        self.checkbox_group = QButtonGroup(self)
        self.checkbox_group.addButton(self.add_zero_checkbox)
        self.checkbox_group.addButton(self.add_pole_checkbox)
        self.checkbox_group.buttonToggled.connect(self.on_checkbox_toggled)
        
        mode_layout.addWidget(self.add_zero_checkbox)
        mode_layout.addWidget(self.add_pole_checkbox)
        mode_group.setLayout(mode_layout)
        
        library_group = QGroupBox("Filter Library")
        library_layout = QVBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.filter_library.keys())

        self.filter_combo.currentIndexChanged.connect(self.load_library_filter)
        library_layout.addWidget(self.filter_combo)
        library_group.setLayout(library_layout)
        
        conjugates_checkbox = QCheckBox("Conjugates")
        conjugates_checkbox.toggled.connect(self.toggle_conjugates)
        mode_layout.addWidget(conjugates_checkbox)
        
        allpass_group = QGroupBox("All-Pass Filters")
        allpass_layout = QVBoxLayout()
        allpass_layout.addWidget(QLabel("Library Filter:"))
        self.allpass_combo = QComboBox()
        self.allpass_combo.addItems(self.allpass_library.keys())
        allpass_layout.addWidget(self.allpass_combo)
        
        allpass_layout.addWidget(QLabel("Custom Filter:"))
        self.custom_a_input = QLineEdit()
        self.custom_a_input.setPlaceholderText("Enter 'a' value (e.g., 0.5 or 0.5+0.5j)")
        allpass_layout.addWidget(self.custom_a_input)
        
        button_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_allpass_filter)
        add_filter_btn = QPushButton("Add Filter")
        add_filter_btn.clicked.connect(self.add_allpass_filter)
        
        button_layout.addWidget(preview_btn)
        button_layout.addWidget(add_filter_btn)
        allpass_layout.addLayout(button_layout)

        enable_checkbox = QCheckBox("Enable")
        enable_checkbox.setChecked(True)
        enable_checkbox.stateChanged.connect(self.toggle_allpass_filter)  
        allpass_layout.addWidget(enable_checkbox)

        
        # Active filters table
        self.allpass_table = QTableWidget()
        self.allpass_table.setColumnCount(3)  
        self.allpass_table.setHorizontalHeaderLabels(["a", "Enabled", "Remove"])
        
        allpass_group.setLayout(allpass_layout)
       
        self.cascadeButton=QPushButton("Export Diagram")
        self.cascadeButton.clicked.connect(self.export_filter_diagrams)

       

       
        left_layout.addWidget(mode_group)
        left_layout.addStretch()

        left_layout.addWidget(allpass_group)
        left_layout.addStretch()
        left_layout.addWidget(library_group)
        left_layout.addStretch()

        left_layout.addWidget(self.cascadeButton)
     
        left_layout.addStretch()
        
        left_widget.setLayout(left_layout)
        
        return left_widget


    def create_center_panel(self):
        center_widget = QWidget()
        center_layout = QVBoxLayout()
        
        # Z-Plane Plot
        plot_group = QGroupBox("Z-Plane")
        plot_layout = QVBoxLayout()
        
        self.z_plane_plot = pg.PlotWidget()
        self.z_plane_plot.setAspectLocked()
        self.z_plane_plot.showGrid(x=True, y=True)
        self.setup_unit_circle()
        self.z_plane_plot.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.z_plane_plot.setMouseEnabled(True, True)
       

        self.z_plane_plot.setBackground('w')  
        self.z_plane_plot.getAxis('bottom').setPen('k')  
        self.z_plane_plot.getAxis('left').setPen('k')   
        self.z_plane_plot.showGrid(x=True, y=True, alpha=0.5)

       

        
        plot_layout.addWidget(self.z_plane_plot)
        plot_group.setLayout(plot_layout)
        
        center_layout.addWidget(plot_group)
        center_widget.setLayout(center_layout)
        return center_widget

    def create_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Magnitude Response
        magnitude_group = QGroupBox("Magnitude Response")
        magnitude_layout = QVBoxLayout()
        self.magnitude_plot = pg.PlotWidget()
        self.magnitude_plot.setBackground('w')
        self.magnitude_plot.getAxis('bottom').setPen('k')
        self.magnitude_plot.getAxis('left').setPen('k')
        self.magnitude_plot.showGrid(x=True, y=True, alpha=0.5)
        
        
      
        self.magnitude_plot.setLabel('left', 'Magnitude (dB)')
        self.magnitude_plot.setLabel('bottom', 'Frequency (Hz)')
        magnitude_layout.addWidget(self.magnitude_plot)
        magnitude_group.setLayout(magnitude_layout)
        
        # Phase Response
        phase_group = QGroupBox("Phase Response")
        phase_layout = QVBoxLayout()
        self.phase_plot = pg.PlotWidget()
        
        # Phase Plot
        self.phase_plot.setBackground('w')
        self.phase_plot.getAxis('bottom').setPen('k')
        self.phase_plot.getAxis('left').setPen('k')
        self.phase_plot.showGrid(x=True, y=True, alpha=0.5)
        self.phase_plot.setLabel('left', 'Phase (radians)')
        self.phase_plot.setLabel('bottom', 'Frequency (Hz)')
        phase_layout.addWidget(self.phase_plot)
        phase_group.setLayout(phase_layout)
        
        right_layout.addWidget(magnitude_group)
        right_layout.addWidget(phase_group)
        right_widget.setLayout(right_layout)
        return right_widget



class AllPassPreviewDialog(QDialog):
    def __init__(self, a, parent=None):
        super().__init__(parent)
        self.a = a
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"All-Pass Filter Preview (a={self.a})")
        layout = QVBoxLayout()
        
        z_plot = pg.PlotWidget()
        z_plot.setBackground('w')
        z_plot.setAspectLocked(True)
        z_plot.showGrid(x=True, y=True)
        
        theta = np.linspace(0, 2*np.pi, 100)
        z_plot.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen('b', width=2))
        
        zero = 1/np.conj(self.a)
        z_plot.plot([zero.real], [zero.imag], pen=None, symbol='o', 
                   symbolBrush='b', symbolSize=10)
        z_plot.plot([self.a.real], [self.a.imag], pen=None, symbol='x', 
                   symbolBrush='r', symbolSize=10)
        
        phase_plot = pg.PlotWidget()
        phase_plot.setBackground('w')
        w, h = freqz([zero], [self.a])
        phase = np.unwrap(np.angle(h))
        phase_plot.plot(w, phase, pen=pg.mkPen('r', width=2))
        phase_plot.setLabel('left', 'Phase (radians)')
        phase_plot.setLabel('bottom', 'Frequency')
        
        layout.addWidget(QLabel("Zero-Pole Plot"))
        layout.addWidget(z_plot)
       
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)