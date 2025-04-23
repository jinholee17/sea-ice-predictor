
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
    variables = ['var34', 'var134', 'var228', 'var147', 'var146']
    print(f"Processing variables: {variables}")
    
    extracted_files = []
    
    for var in variables:
        print(f"Processing variable: {var}")
        
        # Select the variable
        temp_grib = os.path.join(output_dir, f"{var}_temp.grib")
        cmd = ["cdo", "selname," + var, grib_file, temp_grib]
        try:
            subprocess.run(cmd, check=True)
            
            # Map data to Lambert grid 
            lambert_grib = os.path.join(output_dir, f"{var}_lambert.grib")
            cmd = ["cdo", "remapbil,lambert_grid.txt", grib_file, lambert_grib]
            subprocess.run(cmd, check=True)

            
            # Add to the list of extracted files
            extracted_files.append((var, lambert_grib))
            
            # Clean up the temp file
            os.remove(temp_grib)
            
            print(f"Created {lambert_grib}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error processing {var}: {e}")
    
    return extracted_files

def read_grib_files(grib_files): 
    """
    Use CDO to read grib files and extract data
    """
    combined_data = {}
    
    for var_name, grib_file in grib_files:
        try:
            # Use CDO to get spatial data at each time step
            cmd = ["cdo", "output", grib_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # 
            lines = result.stdout.strip().split('\n')
            values = []
            
            # Parse the timestamps with a more robust approach
            for ts in lines:
                try:
                    # Handle various possible formats
                    # First, try to extract just the date portion if there's a T separator
                    if 'T' in ts:
                        date_part = ts.split('T')[0]
                    else:
                        date_part = ts.split()[0]  # In case it's space-separated
                    
                    # Now parse the date part
                    date_parts = date_part.split('-')
                    year = int(date_parts[0])
                    month = int(date_parts[1])
                    day = int(date_parts[2])
                    
                    dates.append(datetime(year, month, day))
                except Exception as e:
                    print(f"Error parsing timestamp {ts}: {e}")
            
            # Use CDO to extract the actual values
            cmd = ["cdo", "outputf,%f", grib_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Process the output to get values
            values_str = result.stdout.strip().split()
            values = []
            
            # Parse the values
            for val_str in values_str:
                try:
                    value = float(val_str)
                    values.append(value)
                except ValueError as e:
                    print(f"Error parsing value {val_str}: {e}")
            
            # Make sure dates and values have the same length
            if len(dates) != len(values):
                print(f"Warning: Number of dates ({len(dates)}) does not match number of values ({len(values)}) for {var_name}")
                
                # If we have dates but not the right number, try an alternative approach
                if len(dates) == 0 and len(values) > 0:
                    print(f"Using alternative approach to get dates for {var_name}")
                    
                    # Try to infer the dates from the meta information
                    cmd = ["cdo", "showdate", grib_file]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    
                    date_strs = result.stdout.strip().split()
                    alternative_dates = []
                    
                    for date_str in date_strs:
                        try:
                            # Format is typically "YYYYMMDD"
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            alternative_dates.append(datetime(year, month, day))
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing date {date_str}: {e}")
                    
                    if len(alternative_dates) == len(values):
                        dates = alternative_dates
                        print(f"Successfully got {len(dates)} dates using alternative method")
                    else:
                        print(f"Alternative method yielded {len(alternative_dates)} dates, still doesn't match values")
                        
                        # As a last resort, create synthetic dates
                        if len(values) > 0:
                            print(f"Creating synthetic dates for {var_name}")
                            # Create a date for each month from 1979-01 to end
                            synthetic_dates = []
                            for i in range(len(values)):
                                year = 1979 + (i // 12)
                                month = (i % 12) + 1
                                synthetic_dates.append(datetime(year, month, 1))
                            
                            dates = synthetic_dates
                            print(f"Created {len(dates)} synthetic dates")
            
            # Store the data
            combined_data[var_name] = {
                'dates': dates,
                'values': values
            }
            
            print(f"Extracted {len(dates)} time points for {var_name}")
            
        except Exception as e:
            print(f"Error processing {var_name}: {e}")
    
    return combined_data
            

def combine_era5_osisaf(): 
    return 

def main(): 
    return 