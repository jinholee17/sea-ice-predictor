import numpy as np
import pywt

def discrete_wave_transform_2d(data):
    # perform spatial discrete wavelet trasnforms on 3D data 
    T, H, W = data.shape
    cA_list, cH_list, cV_list, cD_list = [], [], [], []
    for t in range(T):
        cA, (cH, cV, cD) = pywt.dwt2(data[t], wavelet='db4')
        cA_list.append(cA)
        cH_list.append(cH)
        cV_list.append(cV)
        cD_list.append(cD)
    return np.stack(cA_list), np.stack(cH_list), np.stack(cV_list), np.stack(cD_list)
  

def discrete_wave_transform_1d(data): 
    # perform temporal discrete wavelet transforms on 3D data 
    T, H, W  = data.shape 
    # iterate over each pixel to perform wavelet transform 
    for h in range(H):
        for w in range(W): 
            pixel_slice = data[:, h, w]
            coeffs = pywt.dwt(pixel_slice, wavelet='db4')
            cA, cD = coeffs 
    return np.stack(cA), np.stack(cD)

def transform(data): 
    # for each 12 month chunk (slice when i know how)

    spatial_dwt = discrete_wave_transform_2d(data)
    temporal_dwt = discrete_wave_transform_1d(data)
    return np.stack([spatial_dwt, temporal_dwt], axis=0)