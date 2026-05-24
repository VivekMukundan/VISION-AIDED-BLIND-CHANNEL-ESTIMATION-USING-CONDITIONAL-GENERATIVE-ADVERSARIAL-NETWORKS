import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate_advanced_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Advanced CGAN for Evaluation...")

    # 1. Initialize the new Advanced Generator
    generator = AdvancedGenerator().to(device)
    model_path = "/content/drive/MyDrive/dataset/processed_data/trained_generator_v2.pth"

    # 2. Load the newly trained weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Successfully loaded trained_generator_v2.pth!")

    # 3. Grab one random batch from our FastSDRDataset loader
    # (Assumes 'test_loader' is still in your Colab memory from Step 2)
    real_imgs, real_rfs = next(iter(test_loader))

    # Isolate just the first sample in the batch
    real_img = real_imgs[0:1].to(device)
    real_rf_np = real_rfs[0].numpy() # Shape: (2, 1024)

    # 4. Generate the Blind Channel Prediction
    with torch.no_grad():
        fake_rf = generator(real_img).cpu().squeeze()
        fake_rf_np = fake_rf.numpy() # Shape: (2, 1024)

    # ==========================================
    # METRICS CALCULATION
    # ==========================================
    # NMSE (Normalized Mean Square Error)
    mse = np.sum((real_rf_np - fake_rf_np)**2)
    power_real = np.sum(real_rf_np**2)
    nmse_db = 10 * np.log10(mse / power_real)

    # BER (Hard Decision QPSK Demodulation)
    real_bits = (real_rf_np > 0).astype(int)
    fake_bits = (fake_rf_np > 0).astype(int)
    ber = np.mean(real_bits != fake_bits) * 100

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Advanced 1D-Conv CGAN Evaluation\nNMSE: {nmse_db:.2f} dB | BER: {ber:.2f}%", fontsize=16)

    # Subplot 1: I-Channel
    ax1.plot(real_rf_np[0], label='Real I', color='blue', alpha=0.5, linewidth=1.5)
    ax1.plot(fake_rf_np[0], label='Generated I', color='cyan', linestyle='--', alpha=1.0, linewidth=1.5)
    ax1.set_title("In-Phase (I) Component")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Q-Channel
    ax2.plot(real_rf_np[1], label='Real Q', color='darkorange', alpha=0.5, linewidth=1.5)
    ax2.plot(fake_rf_np[1], label='Generated Q', color='red', linestyle='--', alpha=1.0, linewidth=1.5)
    ax2.set_title("Quadrature (Q) Component")
    ax2.set_xlabel("Sample Index (0-1024)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Execute the evaluation
evaluate_advanced_model()
