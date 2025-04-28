import os
import glob
import xarray as xr
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import subprocess
from datetime import datetime

def process_2D_image_data():
    """
    Requires .nc files from the OSISAF dataset
    Processes them into .npy files
    """
    data_dir = "data/SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009/OSISAF-GLO-SEAICE_CONC_TIMESERIES-NH-LA-OBS_202003"
    start_year = 1989
    end_year = 2022
    resize_shape = (128, 128)
    sequence_length = 5
    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    processed_frames = []

    print("Processing files...")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_dir = os.path.join(data_dir, str(year), f"{month:02d}")
            if not os.path.exists(month_dir):
                continue
            all_files = sorted(glob.glob(os.path.join(month_dir, "*.nc")))

            for f in all_files:
                try:
                    ds = xr.open_dataset(f)
                    date = str(ds.time.values[0])[:10]
                    print(f"Processing: {f}")
                    conc = ds["ice_conc"].isel(time=0).values.astype(np.float32)
                    conc[conc < 0] = np.nan
                    conc[conc > 100] = np.nan
                    conc = np.nan_to_num(conc / 100.0)
                    resized = resize(conc, resize_shape, anti_aliasing=True)
                    processed_frames.append(resized)
                except Exception as e:
                    print(f"Error reading {f}: {e}")

    X = np.stack(processed_frames)
    np.save(os.path.join(output_dir, "all_frames.npy"), X)
    print(f"Saved {len(X)} preprocessed frames.")

    # create input/output sequences for CNN training
    X_seq = []
    y_seq = []

    print("Creating sequences...")
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])   
        y_seq.append(X[i+sequence_length]) 

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_seq = X_seq[..., np.newaxis]
    y_seq = y_seq[..., np.newaxis]
    np.save(os.path.join(output_dir, "X_train.npy"), X_seq)
    np.save(os.path.join(output_dir, "y_train.npy"), y_seq)

    print(f"Saved training data: X_train shape = {X_seq.shape}, y_train shape = {y_seq.shape}")

def extract_era5_variables(grib_file, output_dir):
    """
    Extract all variables from ERA5 GRIB file to separate grib files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    variables = ['sst', 'sp', 'tp', 'slhf', 'sshf']
    print(f"Processing variables: {variables}")
    
    extracted_files = []
    
    for var in variables:
        print(f"Processing variable: {var}")
        
        temp_grib = os.path.join(output_dir, f"{var}_2D.grib").replace("\\", "/")
        print(temp_grib)
        print(grib_file)
        cdo_path = r"C:\Users\dhlee\miniconda3\envs\csci1470\Library\bin\cdo.exe"
        cmd = [cdo_path, "selname," + var, grib_file, temp_grib]

        try:
            subprocess.run(cmd, check=True)
            
            remapped_file = os.path.join(output_dir, f"{var}_2D.grib")
            result = subprocess.run(["cdo", "-remapbil,r128x128.txt", temp_grib , remapped_file])
            if result.returncode != 0:
                print(f"Error remapping to Lambert for {var}: {result.stderr}")
                continue  
            extracted_files.append((var, remapped_file))
            
            os.remove(temp_grib)
            
            print(f"Created {remapped_file}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error processing {var}: {e}")
    
    return extracted_files


def read_grib_data(grib_files):
    """
    Read GRIB data using CDO, return 2D arrays of shape [dim, dim] per year-month.
    """
    all_vars_data = []
    invalid_year_month = [(2010, 11), (2011, 1), (2011, 11), (2012, 1), (2012, 11), (2013, 1), (2013, 11), (2020, 3), (2020, 11)]
    for var_name, grib_file in grib_files:
        try:
            cmd = ["cdo", "outputtab,date,lat,lon,value", grib_file]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error executing CDO command: {result.stderr}")
                continue

            lines = result.stdout.strip().splitlines()
            data = {} 
            curr_year_month = None

            for line in lines:
                
                if any(keyword in line for keyword in ['cdo', 'Processed', 'variable', 'timesteps', '[', 'MB', 's]']):
                    continue
                parts = line.strip().split()
                if len(parts) < 4 or parts[0] == '#':
                    continue
                date_str, lat, lon, value = parts[-4:]
                value = float(value)
                if value == '-9e+33':
                    continue
                date = datetime.strptime(date_str, "%Y-%m-%d")
                year_month = (date.year, date.month)
                if date.year < 1989 or date.year >= 2021 or year_month in invalid_year_month:
                    continue 
                if curr_year_month and curr_year_month != year_month:
                    assert len(data[curr_year_month]) == 128 * 128
                    flattened = data[curr_year_month]
                    flattened = np.asarray(flattened, dtype=float).flatten()
                    data[curr_year_month] = flattened.reshape((128, 128))
                if year_month not in data: 
                    data[year_month] = []
                data[year_month].append(value)
                curr_year_month = year_month

            if data[curr_year_month]:
                flattened = data[curr_year_month]
                data[curr_year_month] = np.array(flattened).reshape((128, 128))
            stacked_data = np.stack([data[date] for date in sorted(data.keys())], axis=0)
            all_vars_data.append(stacked_data)
            print("Finished processing ", var_name)

        except Exception as e:
            print(f"Error processing {var_name}: {e}")
            continue
    return np.array(all_vars_data)


def save_2d_array(data, var_name):
    """
    Save the accumulated x/y/value data into a 2D numpy array file.
    """
    file_name = f"ERA5_data"
    np.save(f"{file_name}_{var_name}", data)
    print(f"Saved {var_name}")

def main(): 
    grib_file = "data/8ebcad7553eb3102c9d0cde229ef4d25.grib"
    output_dir = "2D_extracted_era5"
    print("Extracting ERA5 variables...")
    variables = ['sst', 'sp', 'tp', 'slhf', 'sshf']
    extracted_files = extract_era5_variables(grib_file, output_dir)
    print("Extracting data from grib files...")
    era5_data = read_grib_data(extracted_files)
    print("done!")
    print(era5_data.shape)
    for i, arr in enumerate(era5_data): 
        save_2d_array(arr, variables[i])
    process_2D_image_data()

    era5_data = np.array(era5_data)  # shape: (5, 408, 128, 128)
    return 

if __name__ == "__main__":
    main()