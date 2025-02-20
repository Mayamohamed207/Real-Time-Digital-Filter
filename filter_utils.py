from PyQt5.QtWidgets import QFileDialog

def save_filter(window):
    filename, _ = QFileDialog.getSaveFileName(window, "Save Filter", "", "CSV Files (*.csv)")
    if filename:
        with open(filename, 'w') as f:
            f.write("Zeros,Poles\n")
            for zero, pole in zip(window.zeros, window.poles):
                f.write(f"{zero},{pole}\n")

def load_filter(window):
    filename, _ = QFileDialog.getOpenFileName(window, "Load Filter", "", "CSV Files (*.csv)")
    if filename:
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  
            window.zeros = []
            window.poles = []
            for line in lines:
                zero, pole = line.strip().split(',')
                window.zeros.append(complex(zero))
                window.poles.append(complex(pole))
        window.design_tab.plot_z_plane()