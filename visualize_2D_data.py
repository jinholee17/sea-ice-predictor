import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_nc():
    # Load your file
    #Variables on the dataset include ['time', 'y', 'x', 'lat', 'lon', 'crs', 'conc', 'confidence', 'status_flag']
    file_path = "SEAICE_ARC_PHY_AUTO_L4_NRT_011_015/cmems_obs-si_arc_phy-siconc_nrt_L4-auto_P1D_202105/2023/01/multisensorSeaIce_202301010600.nc"  # replace with your actual filename
    ds = xr.open_dataset(file_path)

    # Preview what's inside
    print(ds)

    # Extract data
    ice = ds["conc"] / 100.0  # convert from 0–100 to fraction
    lons = ds["lon"]
    lats = ds["lat"]

    # Use first timestep if multiple
    if "time" in ice.dims:
        ice = ice.isel(time=0)

    # Plot with Cartopy
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

    # Add land and coastlines
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
    ax.coastlines()

    # Plot sea ice
    ice_plot = ax.pcolormesh(lons, lats, ice * 100, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(ice_plot, label="Sea Ice Concentration (%)", shrink=0.7)

    plt.title("Sea Ice Concentration")
    plt.show()

import numpy as np

def visualize_training_data():
    X_train = np.load("preprocessed/X_train.npy")  # shape: (N, seq_len, H, W, 1)
    y_train = np.load("preprocessed/y_train.npy")  # shape: (N, H, W, 1)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    sample_indices = [0, 10, 20]  # or use random indices

    for idx in sample_indices:
        fig, axes = plt.subplots(1, X_train.shape[1] + 1, figsize=(15, 3))
        fig.suptitle(f"Sample {idx}: Input sequence + Target", fontsize=14)

        # Plot input sequence
        for t in range(X_train.shape[1]):
            axes[t].imshow(X_train[idx, t, :, :, 0], cmap="Blues", vmin=0, vmax=1)
            axes[t].set_title(f"t-{X_train.shape[1] - t}")
            axes[t].axis("off")

        # Plot target
        axes[-1].imshow(y_train[idx, :, :, 0], cmap="Oranges", vmin=0, vmax=1)
        axes[-1].set_title("Target (t+1)")
        axes[-1].axis("off")

        plt.tight_layout()
        plt.show()
        
visualize_training_data()