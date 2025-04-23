## install using command: pip install motuclient
import copernicusmarine 
## see documentation here: https://help.marine.copernicus.eu/en/articles/8286883-copernicus-marine-toolbox-api-get-original-files

import calendar
import os

# dataset_id = "cmems_obs-si_arc_phy-siconc_nrt_L4-auto_P1D"
dataset_id = "OSISAF-GLO-SEAICE_CONC_TIMESERIES-NH-LA-OBS"
#year = 2023
temp_all_files = "temp_full_list.txt"
temp_one_file = "temp_single_file.txt"

for year in range (1989, 2023):
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filter_pattern = f"*{year}{month_str}*"
        print(f"\nchecking {calendar.month_name[month]} {year}...")

        # creates full file list for the month
        try:
            copernicusmarine.get(
                dataset_id=dataset_id,
                filter=filter_pattern,
                create_file_list=temp_all_files
            )
        except Exception as e:
            print(f"Failed to list files: {e}")
            continue

        if not os.path.exists(temp_all_files):
            print("File list not created.")
            continue

        with open(temp_all_files, "r") as f:
            lines = f.readlines()

        if not lines:
            print("No files found for this month.")
            os.remove(temp_all_files)
            continue

        # writes only the first file into a new temp file
        first_file = lines[0].strip()
        with open(temp_one_file, "w") as f:
            f.write(first_file + "\n")

        print(f"will download: {first_file}")
        try:
            copernicusmarine.get(
                dataset_id=dataset_id,
                file_list=temp_one_file
            )
        except Exception as e:
            print(f" Download failed: {e}")

        # Clean up
        os.remove(temp_all_files)
        os.remove(temp_one_file)
