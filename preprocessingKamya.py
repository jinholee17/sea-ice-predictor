
import subprocess
import os
import numpy as np
import pickle
from datetime import datetime
#Reused same logic from preprocessing1D.py 
def extract_era5_variables(grib_file, output_dir):
    """
    Extract all variables from ERA5 GRIB file to separate grib files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables from the file
    variables = ['sst', 'sp', 'tp', 'slhf', 'sshf']
    print(f"Processing variables: {variables}")
    
    extracted_files = []
    
    for var in variables:
        print(f"Processing variable: {var}")
        
        # Select the variable
        temp_grib = os.path.join(output_dir, f"{var}_temp.grib")
        cmd = ["cdo", "selname," + var, grib_file, temp_grib]
        try:
            subprocess.run(cmd, check=True)
            
            #Map data to 128 x 128 Lambert grid 
            remapped_file = os.path.join(output_dir, f"{var}_2D.grib")
            result = subprocess.run(["cdo", "-remapbil,r128x128.txt", temp_grib , remapped_file])
            if result.returncode != 0:
                print(f"Error remapping to Lambert for {var}: {result.stderr}")
                continue  # Skip this variable if remapping failed
            #Add to the list of extracted files
            extracted_files.append((var, remapped_file))
            
            # Clean up the temp file
            os.remove(temp_grib)
            
            print(f"Created {remapped_file}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error processing {var}: {e}")
    
    return extracted_files

# def latlon_to_laea(lat, lon):
#     """
#     Convert lat-lon coordinates to Lambert Azimuthal Equal Area 
#     """
#     lat0 = 90
#     lon0 = 0 
#     earth = 6371000
#     x = 0 
#     y = 0 
#     # Convert degrees to radians
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     lat0_rad = np.radians(lat0)
#     lon0_rad = np.radians(lon0)

#     # following formula for conversion
#     k = np.sqrt(2 / (1 + np.sin(lat0_rad) * np.sin(lat_rad) +
#                        np.cos(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad)))

#     # Calculate new coordinates 
#     x = earth * k * np.cos(lat_rad) * np.sin(lon_rad - lon0_rad)
#     y = earth * k * (np.cos(lat0_rad) * np.sin(lat_rad) -
#                       np.sin(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad))
#     return x, y


def read_grib_data(grib_files):
    """
    Read GRIB data using CDO, return 2D arrays of shape [dim, dim] per year-month.
    """
    all_vars_data = []
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
                # TODO figure out how to get the lat lon to convert and inlcude in data 
                #Skip data points 
                if value == '-9e+33':
                    continue
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date.year < 1989 or date.year >= 2023: 
                    continue 
                year_month = (date.year, date.month)
                if curr_year_month and curr_year_month != year_month:
                    assert len(data[curr_year_month]) == 128 * 128
                    flattened = data[curr_year_month]
                    flattened = np.asarray(flattened, dtype=float).flatten()
                    data[curr_year_month] = flattened.reshape((128, 128))
                if year_month not in data: 
                    data[year_month] = []
                data[year_month].append(value)
                curr_year_month = year_month

            # Save final month
            if data[year_month]:
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

def combine_era5_osisaf(era5_data, osisaf_data):
    # osisaf data has shape [380, 128, 128]
    # ensure that both values passed in are 
    # make the era5 data have same shape.... 
    # map lat/lon to osisaf 
    print(era5_data.shape)
    print(osisaf_data.shape)
    combined_data = np.stack([era5_data, osisaf_data], axis = 0)
    return combined_data 

def main(): 

    # grib_file = "data/8ebcad7553eb3102c9d0cde229ef4d25.grib"
    # output_dir = "./2D_extracted_era5"
    # print("Extracting ERA5 variables...")
    # extracted_files = extract_era5_variables(grib_file, output_dir)
    variables = ['sst', 'sp', 'tp', 'slhf', 'sshf']
    extracted_files = [("slhf", "2D_extracted_era5/slhf_2D.grib"), ('sp',"2D_extracted_era5/sp_2D.grib"), 
                       ('sshf', "2D_extracted_era5/sshf_2D.grib"), ('sst', "2D_extracted_era5/sst_2D.grib"),
                       ('tp', "2D_extracted_era5/tp_2D.grib")]
    print("done!")
    print("Extracting data from grib files...")

    extracted_data = read_grib_data(extracted_files)
    print("done!")
    print(extracted_data.shape)
    for i, arr in enumerate(extracted_data): 
        save_2d_array(arr, variables[i])
    return 

if __name__ == "__main__":
    main()