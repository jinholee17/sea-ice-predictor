import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
