import os
import glob
import xarray as xr
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

data_dir = "data/ice_concentration"
start_year = 1989
end_year = 2022
resize_shape = (128, 128)  # Resize each image to 128x128 for CNN
sequence_length = 5
output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess each daily file
all_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))

processed_frames = []

print("Processing files...")
for f in tqdm(all_files):
    try:
        ds = xr.open_dataset(f)
        date = str(ds.time.values[0])[:10]  # e.g. '1995-03-12'
        year = int(date[:4])

        print(f"Processing: {f}")
        conc = ds["conc"].isel(time=0).values.astype(np.float32)
        conc[conc < 0] = np.nan
        conc[conc > 100] = np.nan
        conc = np.nan_to_num(conc / 100.0)
        resized = resize(conc, resize_shape, anti_aliasing=True)
        processed_frames.append(resized)

    except Exception as e:
        print(f"Error reading {f}: {e}")


X = np.stack(processed_frames)  # Shape: (N, H, W)
np.save(os.path.join(output_dir, "all_frames.npy"), X)
print(f"Saved {len(X)} preprocessed frames.")

# Create input/output sequences for CNN training
X_seq = []
y_seq = []

print("Creating sequences...")
for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])     # shape: (seq_len, H, W)
    y_seq.append(X[i+sequence_length])       # shape: (H, W)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Add channel dimension for CNN: (N, seq_len, H, W, 1)
X_seq = X_seq[..., np.newaxis]
y_seq = y_seq[..., np.newaxis]

# Save as .npy
np.save(os.path.join(output_dir, "X_train.npy"), X_seq)
np.save(os.path.join(output_dir, "y_train.npy"), y_seq)

print(f"Saved training data: X_train shape = {X_seq.shape}, y_train shape = {y_seq.shape}")
