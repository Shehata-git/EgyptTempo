import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ClimateDataset(Dataset):
    """
    Custom PyTorch Dataset for lazy-loading yearly climate tensors.
    Optimized for Google Colab RAM constraints.
    """
    def __init__(self, data_dir="/content/data"):
        self.data_dir = data_dir
        # Identify all X and y files
        self.x_files = sorted(glob.glob(os.path.join(data_dir, "X_*.npy")))
        self.y_files = sorted(glob.glob(os.path.join(data_dir, "y_*.npy")))
        
        if not self.x_files:
            raise FileNotFoundError(f"No X_*.npy files found in {data_dir}")

        self.samples_per_file = []
        self.cumulative_indices = [0]
        
        print("Initializing ClimateDataset and scanning file headers...")
        for x_file in self.x_files:
            # Open with mmap_mode='r' to read shape without loading full data into RAM
            data_mmap = np.load(x_file, mmap_mode='r')
            num_samples = data_mmap.shape[0]
            self.samples_per_file.append(num_samples)
            self.cumulative_indices.append(self.cumulative_indices[-1] + num_samples)
            
        self.total_samples = self.cumulative_indices[-1]
        print(f"Dataset initialized with {self.total_samples} total samples across {len(self.x_files)} years.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        # Binary search or simple loop to find the correct file
        # For ~10 files, a simple loop is efficient enough
        file_idx = 0
        for i in range(len(self.cumulative_indices) - 1):
            if self.cumulative_indices[i] <= idx < self.cumulative_indices[i+1]:
                file_idx = i
                break
        
        # Calculate index within that specific file
        internal_idx = idx - self.cumulative_indices[file_idx]
        
        # Lazy-load only the required sample using memory mapping
        X_mmap = np.load(self.x_files[file_idx], mmap_mode='r')
        y_mmap = np.load(self.y_files[file_idx], mmap_mode='r')
        
        # Extract and convert to Torch tensors
        # .copy() ensures the memory is decoupled from the mmap file
        X_tensor = torch.from_numpy(X_mmap[internal_idx].copy()).float()
        y_tensor = torch.from_numpy(y_mmap[internal_idx].copy()).float()
        
        # Permute if needed (e.g., if ConvLSTM expects Channels-first: Batch, Channel, Time, H, W)
        # Current shape: (Time, H, W, Channel) -> (14, 200, 260, 1)
        # For Conv3D/ConvLSTM usually (Channel, Time, H, W) is preferred:
        # X_tensor = X_tensor.permute(3, 0, 1, 2) 
        
        return X_tensor, y_tensor

def get_dataloader(data_dir="/content/data", batch_size=16, shuffle=True, num_workers=2):
    """
    Returns a PyTorch DataLoader configured for the ClimateDataset.
    """
    dataset = ClimateDataset(data_dir=data_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

if __name__ == "__main__":
    # Quick sanity test (expects local data or mock)
    try:
        loader = get_dataloader(data_dir="./local_staging", batch_size=4)
        for X, y in loader:
            print(f"Batch X shape: {X.shape}") # Expected: (4, 14, 200, 260, 1)
            print(f"Batch y shape: {y.shape}") # Expected: (4, 1, 200, 260, 1)
            break
    except Exception as e:
        print(f"Sanity test skipped: {e}")
