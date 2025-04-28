import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
import os

# === Recombine saved ERA5 variable arrays into input tensor ===
def load_and_stack_era5_variables(variable_names, data_dir="."):
    arrays = []
    for var in variable_names:
        path = os.path.join(data_dir, f"ERA5_data_{var}.npy")
        arr = np.load(path)
        arrays.append(arr)
    stacked = np.stack(arrays, axis=0)
    return stacked

def combine_variables(era5_data):
    osisaf_data = np.load("preprocessed/all_frames.npy")  # shape: (380, 128, 128)
    osisaf_data = osisaf_data[:era5_data.shape[1]]  # make them match in time (408)
    osisaf_data = osisaf_data.astype(np.float32)
    print(f"Clipped OSISAF shape: {osisaf_data.shape}")

    combined = np.concatenate([era5_data, osisaf_data[np.newaxis, ...]], axis=0)  # shape: (6, 408, 128, 128)
    np.save("preprocessed/combined_input.npy", combined)

    print(f"Combined input shape: {combined.shape}")

def main():
    combined = np.load("preprocessed/combined_input.npy")  # (6, 375, 128, 128)
    combined = combined.transpose(1, 0, 2, 3)  # (375, 6, 128, 128)
    X = combined[..., np.newaxis]             # (375, 6, 128, 128, 1)

    # Load or create target y of shape (time, 128, 128, 1)
    y = np.load("preprocessed/y_train.npy")

    min_len = min(X.shape[0], y.shape[0])
    X = X[:min_len]
    y = y[:min_len]
    # Normalize for tanh
    X = X * 2 - 1
    y = y * 2 - 1

    input_layer = Input(shape=(6, 128, 128, 1))

    x = Conv3D(32, (3, 3, 3), activation='tanh', padding='same')(input_layer)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='tanh', padding='same')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(128 * 128, activation='tanh')(x)
    output_layer = Reshape((128, 128, 1))(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # train model
    model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1)

    # save model
    model.save("cnn_wavelet_combined_model.h5")
    return

if __name__ == "__main__":
    main()