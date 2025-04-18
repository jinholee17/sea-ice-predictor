import pickle
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

data_file_paths = ["data/N_01_extent_v3.0.csv", "data/N_02_extent_v3.0.csv","data/N_03_extent_v3.0.csv",
                   "data/N_04_extent_v3.0.csv", "data/N_05_extent_v3.0.csv","data/N_06_extent_v3.0.csv",
                   "data/N_07_extent_v3.0.csv","data/N_08_extent_v3.0.csv","data/N_09_extent_v3.0.csv",
                   "data/N_10_extent_v3.0.csv","data/N_11_extent_v3.0.csv","data/N_12_extent_v3.0.csv",]

def load_data(file_paths,region='N',data_type='Goddard'):
    """
    Load data from file 

    file_paths: list with sea ice data files 
        
    Returns: loaded data 
    """ 
    dates = []
    extent_area = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 6:
                    continue
                year, mo, dtype, reg, extent, area = row[0:6]
                if dtype.strip() == data_type and reg.strip() == region:
                    try:
                        date = datetime(int(year), int(mo), 1)
                        extent_val = float(extent)
                        area_val = float(area)
                        dates.append(date)
                        extent_area.append([extent_val, area_val])
                    except ValueError:
                        continue
    combined = list(zip(dates, extent_area))
    combined.sort(key=lambda x: x[0])
    dates_sorted, values_sorted = zip(*combined)

    return list(dates_sorted), np.array(values_sorted)

def normalize_features(ice_data):
    years = ice_data['year']
    months = ice_data['mo']
    extent = ice_data['extent']
    area = ice_data['area']

    time_vals = years + (months - 1) / 12

    features = np.column_stack([extent, area])
    
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        features[:, i] = np.where(np.isnan(features[:, i]), col_means[i], features[:, i])

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    normalized_data = {
        'time': time_vals,
        'year': years,
        'month': months,
        'extent_normalized': normalized_features[:, 0],
        'area_normalized': normalized_features[:, 1],
        'extent_original': extent,
        'area_original': area,
        'scaler': scaler
    }
    
    return normalized_data


def preprocess_data(file_paths): 
    all_data = []
    for file_path in file_paths:
        try:
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1, 
                                 dtype=[('year', 'i4'), ('mo', 'i4'), ('data-type', 'U10'), 
                                       ('region', 'U2'), ('extent', 'f8'), ('area', 'f8')])
            all_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if len(all_data) > 1:
        combined_data = np.concatenate(all_data)
    else:
        combined_data = all_data[0]

    for field in ['extent', 'area']:
        combined_data[field] = np.where(combined_data[field] == -9999, np.nan, combined_data[field])
    
    mask = (combined_data['year'] >= 1989) & (combined_data['year'] <= 2022)
    filtered_data = combined_data[mask]
    
    return filtered_data

def create_pickle(data_file_paths):
    filtered_data = preprocess_data(data_file_paths)
    normalized = normalize_features(filtered_data)
    output_file="data/data.p"
    with open(output_file, 'wb') as pickle_file:
        pickle.dump(normalized, pickle_file)
    print(f'Data has been dumped into data/data.p!')