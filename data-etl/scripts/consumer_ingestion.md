# Data Ingestion Guide for DL Team

This document provides the necessary cells to ingest the processed climate tensors into a Google Colab environment for model training.

## Step 1: Mount Google Drive
Execute the following cell to mount your Drive.
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Local Transfer & Extraction
> [!IMPORTANT]
> **Performance Warning**: Reading training data directly from the `/content/drive/` (FUSE mount) is extremely slow and will bottleneck your GPU. You **must** transfer the data to the local VM disk (`/content/data/`) before starting the DataLoader.

Run the following bash loop to ingest all years:

```bash
# 1. Create a local high-speed directory
mkdir -p /content/data

# 2. Loop through the processed years and transfer from Drive
# Adjust the year range if necessary
for Y in {2014..2024}; do
    echo "Transferring and extracting year $Y..."
    
    # Path on Drive (ensure this matches the producer's output)
    DRIVE_PATH="/content/drive/MyDrive/Climate_Tensors/climate_tensors_$Y.tar.gz"
    
    if [ -f "$DRIVE_PATH" ]; then
        # Copy to local disk
        cp "$DRIVE_PATH" /content/data/
        
        # Extract locally
        tar -xzf "/content/data/climate_tensors_$Y.tar.gz" -C /content/data/
        
        # Remove the tarball to save space
        rm "/content/data/climate_tensors_$Y.tar.gz"
    else
        echo "Warning: Archive for year $Y not found on Drive."
    fi
done

echo "Ingestion complete. Data is located in /content/data/"
```

## Step 3: Verify Data
You should now see `X_<year>.npy` and `y_<year>.npy` files in `/content/data/`.
```bash
ls -lh /content/data/ | head -n 20
```
