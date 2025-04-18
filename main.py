from preprocessing import create_pickle
data_file_paths = ["data/N_01_extent_v3.0.csv", "data/N_02_extent_v3.0.csv","data/N_03_extent_v3.0.csv",
                   "data/N_04_extent_v3.0.csv", "data/N_05_extent_v3.0.csv","data/N_06_extent_v3.0.csv",
                   "data/N_07_extent_v3.0.csv","data/N_08_extent_v3.0.csv","data/N_09_extent_v3.0.csv",
                   "data/N_10_extent_v3.0.csv","data/N_11_extent_v3.0.csv","data/N_12_extent_v3.0.csv",]
def main():
    create_pickle(data_file_paths)
    print("Data processed")

main()