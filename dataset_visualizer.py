import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

def plot_rf_waveform(rf_data_path):
    # 1. Load the RF data
    # Assuming your data is saved as a numpy .npy file of shape (2, 1024)
    # If it is stored differently (like a .csv or .dat), you would load it accordingly.
    rf_payload = np.load(rf_data_path)

    # Isolate the I and Q channels
    I_channel = rf_payload[0, :]
    Q_channel = rf_payload[1, :]

    # 2. Create the Figure and Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Synchronized RF Baseband Waveform (I/Q)", fontsize=16)

    # 3. Plot In-Phase (I) Component
    ax1.plot(I_channel, color='cyan', linewidth=1.5, label='In-Phase (I)')
    ax1.set_title("In-Phase (I) Component", fontsize=12)
    ax1.set_ylabel("Normalized Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # 4. Plot Quadrature (Q) Component
    ax2.plot(Q_channel, color='red', linewidth=1.5, label='Quadrature (Q)')
    ax2.set_title("Quadrature (Q) Component", fontsize=12)
    ax2.set_xlabel("Sample Index (0-1024)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 5. Final Formatting
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leaves room for the main title
    plt.show()

# ==========================================
# HOW TO RUN IT:
# Replace the string below with the actual path to your specific RF sample
# ==========================================
plot_rf_waveform("/content/drive/MyDrive/dataset/processed_data/rf_tensors/sample_062.npy")
