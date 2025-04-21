import subprocess
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def inspect_grib_with_cdo(grib_file):
    """
    Use CDO to inspect a GRIB file and extract information
    """
    try:
        # Run CDO info command
        print(f"Inspecting GRIB file: {grib_file}")
        cmd = ["cdo", "sinfov", grib_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("CDO info output:")
        print(result.stdout)
        
        # Attempt to determine variables
        print("\nAttempting to extract variables...")
        cmd = ["cdo", "showname", grib_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Variables in file:")
        print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running CDO command: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def extract_era5_variables(grib_file, output_dir, variables=None):
    """
    Extract specified variables from ERA5 GRIB file and convert to CSV
    
    variables: List of variable names to extract (e.g., ['sst', 'tp', 'sp', 'slhf', 'sshf'])
              If None, attempt to extract all variables
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If variables not specified, try to determine them first
        if variables is None:
            cmd = ["cdo", "showname", grib_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            variables = result.stdout.strip().split()
            print(f"Detected variables: {variables}")
        
        # For each variable, extract and convert to CSV
        for var in variables:
            print(f"Processing variable: {var}")
            
            # Select the variable
            temp_grib = os.path.join(output_dir, f"{var}_temp.grib")
            cmd = ["cdo", "selname," + var, grib_file, temp_grib]
            subprocess.run(cmd, check=True)
            
            # Get spatial information
            cmd = ["cdo", "griddes", temp_grib]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Grid description for {var}:")
            print(result.stdout)
            
            # Convert to CSV (time series format) by aggregating spatial dimensions
            csv_file = os.path.join(output_dir, f"{var}_timeseries.csv")
            cmd = ["cdo", "fldmean", "-outputtab,date,value", temp_grib, csv_file]
            subprocess.run(cmd, check=True)
            
            # Clean up temp file
            os.remove(temp_grib)
            
            print(f"Extracted {var} to {csv_file}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running CDO command: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def process_extracted_csv_files(csv_dir, output_file):
    """
    Process the extracted ERA5 CSV files and combine them into a single DataFrame
    """
    # Find all CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_timeseries.csv')]
    
    if not csv_files:
        print("No CSV files found")
        return None
    
    # Initialize the combined DataFrame
    combined_df = None
    
    for csv_file in csv_files:
        var_name = csv_file.split('_')[0]  # Extract variable name from filename
        file_path = os.path.join(csv_dir, csv_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['date', var_name])
            
            # Convert date string to datetime
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # If this is the first file, initialize the combined DataFrame
            if combined_df is None:
                combined_df = df
            else:
                # Otherwise, merge with existing DataFrame
                combined_df = pd.merge(combined_df, df, on='date', how='outer')
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if combined_df is not None:
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
        # Save as pickle
        with open(output_file, 'wb') as f:
            pickle.dump(combined_df, f)
        
        print(f"Combined data saved to {output_file}")
        print(f"Data shape: {combined_df.shape}")
        print(f"Data columns: {combined_df.columns.tolist()}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
    
    return None

# Main execution
if __name__ == "__main__":
    grib_file = "/Users/duruhuseyni/Desktop/cs1470/sea-ice-predictor/data/8ebcad7553eb3102c9d0cde229ef4d25.grib"
    output_dir = "./extracted_era5"
    
    # First inspect the file
    success = inspect_grib_with_cdo(grib_file)
    
    if success:
        # Then extract the variables we're interested in (based on your description)
        # Sea surface temperature (sst), total precipitation (tp), 
        # surface pressure (sp), surface latent heat flux (slhf), 
        # and surface sensible heat flux (sshf)
        variables = ['sst', 'tp', 'sp', 'slhf', 'sshf']
        
        # Try with these specific variable names
        print(f"\nAttempting to extract specified variables: {variables}")
        extract_success = extract_era5_variables(grib_file, output_dir, variables)
        
        if not extract_success:
            # If that fails, try with whatever variables are in the file
            print("\nFailed with specified variables. Trying to detect variables automatically...")
            extract_success = extract_era5_variables(grib_file, output_dir, None)
        
        if extract_success:
            # Process the extracted CSV files
            era5_df = process_extracted_csv_files(output_dir, "era5_data.pkl")
            
            if era5_df is not None:
                print("\nSuccessfully processed ERA5 data!")
                print(era5_df.head())