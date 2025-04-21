import subprocess
import os
import numpy as np
import pickle
from datetime import datetime
import re

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
            
            # Calculate spatial mean
            mean_grib = os.path.join(output_dir, f"{var}_mean.grib")
            cmd = ["cdo", "fldmean", temp_grib, mean_grib]
            subprocess.run(cmd, check=True)
            
            # Add to the list of extracted files
            extracted_files.append((var, mean_grib))
            
            # Clean up the temp file
            os.remove(temp_grib)
            
            print(f"Created {mean_grib}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error processing {var}: {e}")
    
    return extracted_files

def read_grib_data(grib_files):
    """
    Use CDO to read grib files and extract data
    """
    combined_data = {}
    
    for var_name, grib_file in grib_files:
        try:
            # Use CDO to get the time values
            cmd = ["cdo", "showtimestamp", grib_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Process the output to get timestamps
            timestamps = result.stdout.strip().split()
            dates = []
            
            # Parse the timestamps with a more robust approach
            for ts in timestamps:
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

def convert_extracted_data_to_monthly(extracted_data):
    """
    Convert the extracted data to monthly time series
    """
    monthly_data = {
        'dates': [],
        'year': [],
        'month': []
    }
    
    # Initialize columns for each variable
    for var in extracted_data:
        monthly_data[var] = []
    
    # First, create a set of all unique year-month combinations
    all_year_months = set()
    
    for var in extracted_data:
        if 'dates' in extracted_data[var] and extracted_data[var]['dates']:
            for date in extracted_data[var]['dates']:
                all_year_months.add((date.year, date.month))
    
    # Sort the year-months chronologically
    all_year_months = sorted(all_year_months)
    
    # For each year-month, compute the average value for each variable
    for year, month in all_year_months:
        monthly_data['year'].append(year)
        monthly_data['month'].append(month)
        monthly_data['dates'].append(datetime(year, month, 1))
        
        for var in extracted_data:
            # Find all values for this variable in this year-month
            values = []
            
            if 'dates' in extracted_data[var] and extracted_data[var]['dates']:
                for i, date in enumerate(extracted_data[var]['dates']):
                    if date.year == year and date.month == month:
                        values.append(extracted_data[var]['values'][i])
            
            # Calculate the average value
            if values:
                avg_value = np.mean(values)
            else:
                avg_value = np.nan
            
            monthly_data[var].append(avg_value)
    
    # Convert to numpy arrays
    for key in monthly_data:
        if key != 'dates':
            monthly_data[key] = np.array(monthly_data[key])
    
    return monthly_data

def read_nsidc_csv(file_paths):
    """
    Read and combine NSIDC CSV files
    """
    combined_data = {
        'year': [],
        'month': [],
        'extent': [],
        'area': []
    }
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                # Skip header
                next(f)
                
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        year = int(parts[0])
                        month = int(parts[1])
                        data_type = parts[2].strip()
                        region = parts[3].strip()
                        
                        # Only include Goddard data and N region as in your reference code
                        if data_type != 'Goddard' or region != 'N':
                            continue
                        
                        # Handle extent and area
                        try:
                            extent = float(parts[4])
                            if extent == -9999:
                                extent = np.nan
                        except ValueError:
                            extent = np.nan
                        
                        try:
                            area = float(parts[5])
                            if area == -9999:
                                area = np.nan
                        except ValueError:
                            area = np.nan
                        
                        combined_data['year'].append(year)
                        combined_data['month'].append(month)
                        combined_data['extent'].append(extent)
                        combined_data['area'].append(area)
            
            print(f"Processed {file_path}")
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not combined_data['year']:
        print("No data was read from NSIDC files")
        return combined_data
    
    # Convert to numpy arrays
    for key in combined_data:
        combined_data[key] = np.array(combined_data[key])
    
    # Add dates
    combined_data['dates'] = [datetime(year, month, 1) for year, month in 
                              zip(combined_data['year'], combined_data['month'])]
    
    # Filter to include only 1989-2022
    mask = (combined_data['year'] >= 1989) & (combined_data['year'] <= 2022)
    for key in combined_data:
        if key == 'dates':
            combined_data[key] = [d for i, d in enumerate(combined_data[key]) if mask[i]]
        else:
            combined_data[key] = combined_data[key][mask]
    
    return combined_data

def merge_datasets(nsidc_data, era5_data):
    """
    Merge NSIDC and ERA5 datasets
    """
    # Handle empty datasets
    if len(nsidc_data.get('year', [])) == 0 or len(era5_data.get('year', [])) == 0:
        print("Cannot merge: One or both datasets are empty")
        return {'year': [], 'month': [], 'dates': [], 'extent': [], 'area': []}
    
    merged_data = {
        'year': [],
        'month': [],
        'dates': [],
        'extent': [],
        'area': []
    }
    
    # Add ERA5 variables to merged data structure
    for key in era5_data:
        if key not in ['year', 'month', 'dates'] and key not in merged_data:
            merged_data[key] = []
    
    # Create year-month tuples for matching
    nsidc_year_month = [(date.year, date.month) for date in nsidc_data['dates']]
    era5_year_month = [(year, month) for year, month in zip(era5_data['year'], era5_data['month'])]
    
    # For each NSIDC data point, find corresponding ERA5 data
    matches_found = 0
    
    for i, (year, month) in enumerate(nsidc_year_month):
        try:
            era5_idx = era5_year_month.index((year, month))
            
            # Found a match, add data to merged dataset
            merged_data['year'].append(year)
            merged_data['month'].append(month)
            merged_data['dates'].append(nsidc_data['dates'][i])
            merged_data['extent'].append(nsidc_data['extent'][i])
            merged_data['area'].append(nsidc_data['area'][i])
            
            # Add ERA5 variables
            for key in era5_data:
                if key not in ['year', 'month', 'dates'] and key in merged_data:
                    merged_data[key].append(era5_data[key][era5_idx])
            
            matches_found += 1
        
        except ValueError:
            # No matching ERA5 data for this date, skip
            pass
    
    print(f"Found {matches_found} matching time points between datasets")
    
    # Check if we merged any data
    if len(merged_data['year']) == 0:
        print("No overlapping data points found between NSIDC and ERA5 datasets")
        return merged_data
    
    # Convert to numpy arrays
    for key in merged_data:
        if key != 'dates':
            merged_data[key] = np.array(merged_data[key])
    
    return merged_data

def normalize_data(data):
    """
    Normalize the dataset using Min-Max scaling
    """
    normalized_data = {}
    
    # Check if data is empty
    if len(data.get('year', [])) == 0:
        print("No data to normalize")
        return data
    
    # Copy non-numeric data
    for key in ['year', 'month', 'dates']:
        if key in data:
            normalized_data[key] = data[key]
    
    # Get numeric columns for normalization
    numeric_keys = [key for key in data if key not in ['year', 'month', 'dates']]
    
    # Store original data
    for key in numeric_keys:
        normalized_data[f"{key}_original"] = data[key]
    
    # Normalize each numeric column
    for key in numeric_keys:
        values = data[key]
        
        # Check if array is empty
        if len(values) == 0:
            normalized_data[f"{key}_normalized"] = np.array([])
            normalized_data[f"{key}_min"] = np.nan
            normalized_data[f"{key}_max"] = np.nan
            continue
        
        # Handle NaN values
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            mean_val = np.nanmean(values)
            values = np.where(nan_mask, mean_val, values)
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Avoid division by zero
        if max_val > min_val:
            norm_values = (values - min_val) / (max_val - min_val)
        else:
            norm_values = np.zeros_like(values)
        
        normalized_data[f"{key}_normalized"] = norm_values
        # Store normalization parameters for later inverse transform
        normalized_data[f"{key}_min"] = min_val
        normalized_data[f"{key}_max"] = max_val
    
    return normalized_data

def prepare_for_lstm(normalized_data, seq_length=12, target='extent'):
    """
    Prepare sequences for LSTM model
    """
    # Check if data is empty
    if len(normalized_data.get('year', [])) == 0:
        print("No data to prepare for LSTM")
        return np.array([]), np.array([]), []
    
    # Get normalized feature columns
    feature_cols = [col for col in normalized_data if col.endswith('_normalized')]
    target_col = f"{target}_normalized"
    
    # Ensure target column exists
    if target_col not in feature_cols:
        print(f"Target column {target_col} not found in normalized data")
        return np.array([]), np.array([]), feature_cols
    
    # Create arrays for X (features) and y (target)
    X = []
    y = []
    
    data_length = len(normalized_data['year'])
    
    # Check if we have enough data for at least one sequence
    if data_length <= seq_length:
        print(f"Not enough data points ({data_length}) for sequence length {seq_length}")
        return np.array([]), np.array([]), feature_cols
    
    # For each possible sequence
    for i in range(data_length - seq_length):
        # Create sequence array
        seq_x = np.zeros((seq_length, len(feature_cols)))
        
        for j, col in enumerate(feature_cols):
            seq_x[:, j] = normalized_data[col][i:i+seq_length]
        
        # Target is the next time step after the sequence
        target_idx = i + seq_length
        seq_y = normalized_data[target_col][target_idx]
        
        X.append(seq_x)
        y.append(seq_y)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, feature_cols

def main():
    # File paths
    grib_file = "/Users/duruhuseyni/Desktop/cs1470/sea-ice-predictor/data/8ebcad7553eb3102c9d0cde229ef4d25.grib"
    output_dir = "./extracted_era5"
    
    # NSIDC file paths - using the correct paths from your reference code
    nsidc_files = [
        "data/N_01_extent_v3.0.csv", "data/N_02_extent_v3.0.csv", "data/N_03_extent_v3.0.csv",
        "data/N_04_extent_v3.0.csv", "data/N_05_extent_v3.0.csv", "data/N_06_extent_v3.0.csv", 
        "data/N_07_extent_v3.0.csv", "data/N_08_extent_v3.0.csv", "data/N_09_extent_v3.0.csv",
        "data/N_10_extent_v3.0.csv", "data/N_11_extent_v3.0.csv", "data/N_12_extent_v3.0.csv"
    ]
    
    # Extract ERA5 variables to separate grib files
    print("Extracting ERA5 variables...")
    extracted_files = extract_era5_variables(grib_file, output_dir)
    
    if not extracted_files:
        print("No variables extracted from GRIB file")
        return
    
    # Read data from the grib files
    print("\nReading data from grib files...")
    extracted_data = read_grib_data(extracted_files)
    
    if not extracted_data:
        print("No data read from grib files")
        return
    
    # Print a sample of dates and values for debugging
    for var in extracted_data:
        print(f"\nSample data for {var}:")
        if 'dates' in extracted_data[var] and extracted_data[var]['dates']:
            num_samples = min(5, len(extracted_data[var]['dates']))
            for i in range(num_samples):
                print(f"  Date: {extracted_data[var]['dates'][i]}, Value: {extracted_data[var]['values'][i]}")
        else:
            print("  No dates available")
    
    # Convert to monthly time series
    print("\nConverting to monthly time series...")
    era5_data = convert_extracted_data_to_monthly(extracted_data)
    
    # Print a sample of the monthly data for debugging
    print("\nSample of monthly data:")
    if era5_data['dates']:
        num_samples = min(5, len(era5_data['dates']))
        for i in range(num_samples):
            print(f"  Date: {era5_data['dates'][i]}")
            for var in extracted_data:
                print(f"    {var}: {era5_data[var][i]}")
    else:
        print("  No monthly data available")
    
    # Verify NSIDC files exist
    existing_nsidc_files = []
    for file_path in nsidc_files:
        if os.path.exists(file_path):
            existing_nsidc_files.append(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    if not existing_nsidc_files:
        print("No NSIDC CSV files found at the specified paths")
        return
    
    print(f"\nProcessing {len(existing_nsidc_files)} NSIDC CSV files...")
    nsidc_data = read_nsidc_csv(existing_nsidc_files)
    
    # Print a sample of the NSIDC data for debugging
    print("\nSample of NSIDC data:")
    if 'dates' in nsidc_data and nsidc_data['dates']:
        num_samples = min(5, len(nsidc_data['dates']))
        for i in range(num_samples):
            print(f"  Date: {nsidc_data['dates'][i]}, Extent: {nsidc_data['extent'][i]}, Area: {nsidc_data['area'][i]}")
    else:
        print("  No NSIDC data available")
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_data = merge_datasets(nsidc_data, era5_data)
    
    print(f"Merged data contains {len(merged_data['dates'])} time points")
    
    # Print a sample of the merged data for debugging
    print("\nSample of merged data:")
    if merged_data['dates']:
        num_samples = min(5, len(merged_data['dates']))
        for i in range(num_samples):
            print(f"  Date: {merged_data['dates'][i]}, Extent: {merged_data['extent'][i]}")
    else:
        print("  No merged data available")
    
    # Normalize data
    print("\nNormalizing data...")
    normalized_data = normalize_data(merged_data)
    
    # Prepare for LSTM
    print("\nPreparing data for LSTM...")
    X, y, feature_cols = prepare_for_lstm(normalized_data)
    
    if X.size > 0:
        print(f"LSTM input shape: {X.shape}, output shape: {y.shape}")
        print(f"Features used: {feature_cols}")
    else:
        print("No LSTM sequences created")
    
    # Save processed data
    output_file = "processed_ice_data.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'feature_cols': feature_cols,
            'normalized_data': normalized_data,
            'variable_mapping': {
                'var34': 'possibly sea surface temperature (SST)',
                'var134': 'possibly surface pressure (SP)',
                'var228': 'possibly total precipitation (TP)',
                'var147': 'possibly surface latent heat flux (SLHF)',
                'var146': 'possibly surface sensible heat flux (SSHF)'
            }
        }, f)
    
    print(f"\nProcessed data saved to {output_file}")

if __name__ == "__main__":
    main()