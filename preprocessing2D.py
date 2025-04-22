## install using command: pip install motuclient
import copernicusmarine 
## uncomment to get 'files.txt' with all file names to download from copernicus
## see documentation here: https://help.marine.copernicus.eu/en/articles/8286883-copernicus-marine-toolbox-api-get-original-files
## copernicusmarine.get(dataset_id="OSISAF-GLO-SEAICE_CONC_TIMESERIES-NH-LA-OBS", create_file_list="files.txt")

# copernicusmarine.get(dataset_id="seaice_arc_nh_polstere_100_daily", filter="*2023*")
# copernicusmarine.subset(
#   dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
#   variables=["uo", "vo"],
#   minimum_longitude=50,
#   maximum_longitude=90,
#   minimum_latitude=0,
#   maximum_latitude=25,
#   start_datetime="2022-06-01T00:00:00",
#   end_datetime="2022-06-29T23:59:59",
#   minimum_depth=0,
#   maximum_depth=30,
#   output_filename = "CMEMS_Indian_currents_Jan2022.nc",
#   output_directory = "copernicus-data"
# )

import copernicusmarine
import calendar
import os

dataset_id = "cmems_obs-si_arc_phy-siconc_nrt_L4-auto_P1D"
year = 2023
temp_all_files = "temp_full_list.txt"
temp_one_file = "temp_single_file.txt"

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
