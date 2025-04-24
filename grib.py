import cfgrib
import pandas as pd
import sys

def grib_to_csv(grib_file, output_csv):
    try:
        ds = cfgrib.open_dataset(grib_file)
        df = ds.to_dataframe().reset_index()
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV to: {output_csv}")
    except Exception as e:
        print(f"Failed to convert GRIB to CSV: {e}")

if __name__ == "__main__":
    grib_file = "data/8ebcad7553eb3102c9d0cde229ef4d25.grib"
    output_csv = "output.csv"
    grib_to_csv(grib_file, output_csv)

