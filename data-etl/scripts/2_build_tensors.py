import argparse
import os
import gc
import numpy as np

def build_tensors(year):
    # Determine base directory
    base_dir = "/content/local_staging"
    if not os.path.exists(base_dir):
        base_dir = "./local_staging"
    
    file_path = os.path.join(base_dir, f"raw_scaled_{year}.npy")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading scaled data for year {year}...")
    data = np.load(file_path)
    
    num_days, rows, cols = data.shape
    lookback = 14
    
    num_samples = num_days - lookback
    if num_samples <= 0:
        print(f"Year {year} has only {num_days} days, which is less than lookback {lookback}. Skipping...")
        return

    print(f"Building tensors for {num_samples} samples...")
    # ConvLSTM Input: (Samples, Time, Rows, Cols, Channels)
    X = np.zeros((num_samples, lookback, rows, cols, 1), dtype=np.float32)
    # ConvLSTM Target: (Samples, 1, Rows, Cols, Channels)
    y = np.zeros((num_samples, 1, rows, cols, 1), dtype=np.float32)
    
    for i in range(num_samples):
        # Slice 14 days for X
        X[i, :, :, :, 0] = data[i : i + lookback]
        # Next day for y
        y[i, 0, :, :, 0] = data[i + lookback]
        
    X_path = os.path.join(base_dir, f"X_{year}.npy")
    y_path = os.path.join(base_dir, f"y_{year}.npy")
    
    print(f"Saving tensors to {X_path} and {y_path}...")
    np.save(X_path, X)
    np.save(y_path, y)
    
    print("Deleting raw scaled file and freeing memory...")
    os.remove(file_path)
    del data, X, y
    gc.collect()
    print(f"Successfully processed year {year}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    args = parser.parse_args()
    build_tensors(args.year)
