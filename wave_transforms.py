import numpy as np
import pywt
from skimage.transform import resize

def discrete_wave_transform_2d(data):
    # perform spatial discrete wavelet trasnforms on 3D data 
    T, H, W = data.shape
    cA_list, cH_list, cV_list, cD_list = [], [], [], []
    for t in range(T):
        cA, (cH, cV, cD) = pywt.dwt2(data[t], wavelet='db4')
        # cA is the approximation coefficients, cH, cV, cD are the detail coefficients
        cA_list.append(cA)
        cH_list.append(cH)
        cV_list.append(cV)
        cD_list.append(cD)
    #return all the coefficients as 4D arrays 
    return np.stack(cA_list), np.stack(cH_list), np.stack(cV_list), np.stack(cD_list)

def discrete_wave_transform_1d(data): 
    T, H, W  = data.shape

    # Determine true output length using a sample transform
    sample_slice = data[:, 0, 0]
    cA_len = len(pywt.dwt(sample_slice, wavelet='db4')[0])

    cA_cube = np.zeros((cA_len, H, W))
    cD_cube = np.zeros((cA_len, H, W))

    # Perform the 1D DWT on each pixel slice
    for h in range(H):
        for w in range(W): 
            pixel_slice = data[:, h, w]
            cA, cD = pywt.dwt(pixel_slice, wavelet='db4')
            cA_cube[:, h, w] = cA
            cD_cube[:, h, w] = cD
    #return coefficients as 3D arrays 
    return cA_cube, cD_cube



def transform(data): 
    # for each 12  month chunk
    #shape should be (6, 380, 128, 128)
    spatial_dwts = [] 
    temporal_dwts = []
    slice_size = 12 
    num_full_slices = data.shape[1] // slice_size
    leftover = data.shape[1] % slice_size
    data_trimmed = data[:, leftover:, :, :] # discarding the first 20 entries
    sliced_array = np.stack([data_trimmed[:, i*slice_size:(i+1)*slice_size, :, :]
    for i in range(num_full_slices)], axis=0)
    # perform the wavelet transform on each year slice 
    for slice in sliced_array: 
        spatial_dwts.append(discrete_wave_transform_2d(slice))
        temporal_dwts.append(discrete_wave_transform_1d(slice))
    # return the transformed data to be inputted into the model
    return spatial_dwts, temporal_dwts

frames = np.load("preprocessed/all_frames.npy")  # shape: (N, H, W)
print("Loaded shape:", frames.shape)

start_year = 1989
end_year = 2022
years = list(range(start_year, end_year + 1))
assert len(years) == 34

sampled = frames[::12][:len(years)]  # (34, H, W)
print(f"Sampled shape (1 per year): {sampled.shape}")
resized_sampled = np.stack([resize(img, (128, 128)) for img in sampled])
resized_sampled = resized_sampled[:32]  # Trim to 32 time steps


spatial_out, temporal_out = transform(resized_sampled)

print(f"Spatial Output shapes: {[s.shape for s in spatial_out]}")
print(f"Temporal Output shapes: {[s.shape for s in temporal_out]}")

#prints the original npy file size:

file_path = "preprocessed/all_frames.npy"

data = np.load(file_path)
print(f"Shape of input data, {file_path}, was: {data.shape}")
