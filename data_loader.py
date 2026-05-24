import os
import math
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from google.colab import drive

# ==========================================
# 1. MOUNT GOOGLE DRIVE & SETUP PATHS
# ==========================================
print("Mounting Google Drive...")
drive.mount('/content/drive')

# IMPORTANT: Ensure this points to the folder containing setup_001, setup_002, etc.
RAW_DATA_DIR = "/content/drive/MyDrive/dataset/dataset"
PROCESSED_DATA_DIR = "/content/drive/MyDrive/dataset/processed_data"

os.makedirs(os.path.join(PROCESSED_DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "rf_tensors"), exist_ok=True)

# ==========================================
# 2. PROCESSING FUNCTIONS
# ==========================================
def process_lidar_to_image(csv_path, save_path):
    df = pd.read_csv(csv_path)
    canvas = np.zeros((256, 256), dtype=np.uint8)
    max_dist = 5000.0

    for _, row in df.iterrows():
        dist = row.get('range', row.get('distance', 0))
        angle_deg = row.get('angle', 0)

        if dist == 0: continue

        angle_rad = math.radians(angle_deg)
        x = int((dist * math.cos(angle_rad) / max_dist) * 128 + 128)
        y = int((dist * math.sin(angle_rad) / max_dist) * 128 + 128)

        if 0 <= x < 256 and 0 <= y < 256:
            canvas[y, x] = 255

    cv2.imwrite(save_path, canvas)

def process_rf_signal(dat_path, save_path):
    signal = np.fromfile(dat_path, dtype=np.complex64)
    amplitude = np.abs(signal)

    start_indices = np.where(amplitude > 0.1)[0]
    if len(start_indices) == 0:
        return False, 0.0

    start_idx = start_indices[0]
    if start_idx + 1024 > len(signal):
        return False, 0.0

    raw_payload = signal[start_idx : start_idx + 1024]
    iq_matrix = np.vstack((np.real(raw_payload), np.imag(raw_payload)))

    np.save(save_path, iq_matrix)
    return True, np.max(np.abs(iq_matrix))

# ==========================================
# 3. THE NESTED FOLDER MATCHER
# ==========================================
def build_dataset_v3():
    print(f"\nScanning for setup folders in: {RAW_DATA_DIR}")
    dataset_records = []
    global_rf_max = 0.0
    valid_samples_count = 0

    # Outer loop: Folders setup_001 to setup_005
    for setup_num in range(1, 6):
        setup_folder_name = f"setup_{setup_num:03d}"

        # Inner loop: Files 01 to 20 inside each folder
        for file_num in range(1, 21):

            # --- THE FIX IS HERE ---
            # lidar_001.csv to lidar_020.csv
            lidar_filename = f"lidar_{file_num:03d}.csv"

            # rf_burst_01.dat to rf_burst_20.dat
            rf_filename = f"rx_burst_{file_num:02d}.dat"
            # -----------------------

            csv_path = os.path.join(RAW_DATA_DIR, setup_folder_name, lidar_filename)
            dat_path = os.path.join(RAW_DATA_DIR, setup_folder_name, rf_filename)

            # Check if files exist
            if not os.path.exists(csv_path):
                print(f"Skipping: Could not find {csv_path}")
                continue
            if not os.path.exists(dat_path):
                print(f"Skipping: Could not find {dat_path}")
                continue

            # Create a continuous global ID (001 to 100) for clean saving
            global_idx = (setup_num - 1) * 20 + file_num
            sample_id = f"sample_{global_idx:03d}"

            # Standardized output paths
            img_save_path = os.path.join(PROCESSED_DATA_DIR, "images", f"{sample_id}.png")
            rf_save_path = os.path.join(PROCESSED_DATA_DIR, "rf_tensors", f"{sample_id}.npy")

            # Process and save
            process_lidar_to_image(csv_path, img_save_path)
            success, local_rf_max = process_rf_signal(dat_path, rf_save_path)

            if success:
                if local_rf_max > global_rf_max:
                    global_rf_max = local_rf_max

                dataset_records.append({
                    "sample_id": sample_id,
                    "image_path": img_save_path,
                    "rf_path": rf_save_path
                })
                valid_samples_count += 1
                print(f"Processed {setup_folder_name} -> {sample_id}...", end="\r")

    print(f"\n\nPreprocessing Complete! Total Valid Samples: {valid_samples_count}")
    print(f"Global RF Maximum Value: {global_rf_max}")

    index_df = pd.DataFrame(dataset_records)
    index_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "dataset_index.csv"), index=False)

    with open(os.path.join(PROCESSED_DATA_DIR, "global_rf_max.txt"), "w") as f:
        f.write(str(global_rf_max))

# Execute the builder
build_dataset_v3()

# ==========================================
# 4. THE OPTIMIZED PYTORCH DATASET
# ==========================================
class FastSDRDataset(Dataset):
    def __init__(self, index_csv, max_rf_file):
        self.data_frame = pd.read_csv(index_csv)
        with open(max_rf_file, 'r') as f:
            self.rf_max = float(f.read())
        if self.rf_max == 0: self.rf_max = 1e-6

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['image_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        rf_path = self.data_frame.iloc[idx]['rf_path']
        rf_payload = torch.tensor(np.load(rf_path), dtype=torch.float32) / self.rf_max

        return image, rf_payload

# Test the Dataset
print("\n--- Testing the DataLoader ---")
index_path = os.path.join(PROCESSED_DATA_DIR, "dataset_index.csv")
max_path = os.path.join(PROCESSED_DATA_DIR, "global_rf_max.txt")
if os.path.exists(index_path):
    test_dataset = FastSDRDataset(index_path, max_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    test_imgs, test_rfs = next(iter(test_loader))
    print(f"Data Loaded Successfully! RF Batch Shape: {test_rfs.shape}")
