import sys
from multiprocessing import Process, Queue
import pyqtgraph.Qt as qt
import pyqtgraph as pg
import numpy as np
import time
import scipy.signal as scipySignal
import pyaudio

decayed_color = '#00FF00'  # Bright green for the decayed plot line
real_time_color = '#0096FF'  # Blue for the real-time plot line
background_color_decayed = '#003300'  # Dark green background for the decayed plot
background_color_real_time = '#FFFFFF'  # White background for the real-time plot
grid_color_decayed = '#FFFFFF'  # White for the grid and axis in the decayed plot
grid_color_real_time = '#FFFFFF'  # White for the grid and axis in the real-time plot

# Audio parameters
RATE = 48000
CHUNK = 2048
NOISE_THRESHOLD = 0.02  # Adjusted threshold to reduce sensitivity to low-level noise
DECAY_RATE = 0.9  # Decay rate to control the speed of the return to zero (0 < DECAY_RATE < 1)

# Initialize PyQtGraph application
app = qt.QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="Erra P. - FFT Graph Comparison")
win.setGeometry(100, 100, 900, 600)

# Create two plots in the window: one with decay and one without decay
decayed_plot = win.addPlot(title="Erra P - FFT Plot with Slow Decay")
decayed_curve = decayed_plot.plot(pen=pg.mkPen(color=decayed_color, width=2.5))  # Bright green line with decay

# Set axis labels and ranges for decayed plot
decayed_plot.setLabel('bottom', 'Frequency', units='Hz')
decayed_plot.setLabel('left', 'Amplitude (With Decay)')
decayed_plot.setXRange(0, 5000, padding=0)  # Set x-axis range from 0 to 5000 Hz
decayed_plot.getAxis('bottom').setTickSpacing(1000, 250)
decayed_plot.getAxis('bottom').setStyle(tickTextOffset=10, autoExpandTextSpace=True)
decayed_plot.getAxis('bottom').setStyle(tickFont=qt.QtGui.QFont("Arial", 10))
decayed_plot.getAxis('bottom').enableAutoSIPrefix(False)

# Set background color for the decayed plot
pg.setConfigOption('background', background_color_decayed)  # Dark green background for the decayed plot
pg.setConfigOption('foreground', grid_color_decayed)  # White for axis and grid lines in decayed plot
decayed_plot.showGrid(x=True, y=True, alpha=0.3)

# Add a second plot (without decay) below the first one
win.nextRow()  # Start a new row in the layout for the second plot
real_time_plot = win.addPlot(title="Erra P. - FFT Plot")
real_time_curve = real_time_plot.plot(pen=pg.mkPen(color=real_time_color, width=2.5))  # Blue line without decay

# Set axis labels and ranges for real-time plot
real_time_plot.setLabel('bottom', 'Frequency', units='Hz')
real_time_plot.setLabel('left', 'Amplitude (No Decay)')
real_time_plot.setXRange(0, 5000, padding=0)  # Set x-axis range from 0 to 5000 Hz
real_time_plot.getAxis('bottom').setTickSpacing(1000, 250)
real_time_plot.getAxis('bottom').setStyle(tickTextOffset=10, autoExpandTextSpace=True)
real_time_plot.getAxis('bottom').setStyle(tickFont=qt.QtGui.QFont("Arial", 10))
real_time_plot.getAxis('bottom').enableAutoSIPrefix(False)

# Set background color for the real-time plot
pg.setConfigOption('background', background_color_real_time)  # White background for the real-time plot
pg.setConfigOption('foreground', grid_color_real_time)  # White for the axis and grid lines
real_time_plot.showGrid(x=True, y=True, alpha=0.5)

def stream(q):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(data, dtype=np.int16)
            q.put(data)
        except Exception as e:
            print(f"Error reading stream: {e}")
        time.sleep(0.01)  # Short delay to prevent CPU overload

def update(q: Queue):
    win.show()
    last_fft = np.zeros(CHUNK // 2)  # Initialize last FFT data array to zero
    
    while True:
        try:
            data = q.get_nowait()
        except:
            app.processEvents()
            time.sleep(0.01)
            continue
        
        # Detrend (no DC hum) and compute FFT
        signal = scipySignal.detrend(data)
        fft = np.abs(np.fft.fft(signal)) * 2 / CHUNK  # Normalizs the FFT
        fft = fft[:len(fft) // 2]  # Keep only the positive frequencies
        
        # Generate frequencies corresponding to the FFT data
        freqs = np.fft.fftfreq(len(signal), 1.0 / RATE)[:len(fft)]
        
        # Apply a noise threshold to filter out low-amplitude fluctuations
        fft[fft < NOISE_THRESHOLD] = 0  # Set values below threshold to zero
        
        # FFT with decay
        decayed_fft = np.maximum(fft, last_fft * DECAY_RATE)
        last_fft = decayed_fft  # Update last_fft for the next frame

        # Dynamically adjust y-axis range based on FFT max amplitude
        max_amplitude = max(np.max(decayed_fft), np.max(fft), 1)
        decayed_plot.setYRange(0, max_amplitude, padding=0)
        real_time_plot.setYRange(0, max_amplitude, padding=0)  # Synchronize y-axis ranges

        # Update both curves with their respective FFT data
        decayed_curve.setData(freqs, decayed_fft)  # Decayed FFT data
        real_time_curve.setData(freqs, fft)        # Real-time FFT data (no decay)

        app.processEvents()  # Keep the GUI responsive

if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=update, args=(q,))
    p2 = Process(target=stream, args=(q,))
    p2.start()
    p1.start()
    p1.join()
    p2.join()
