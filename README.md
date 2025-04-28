# seaicepredictor
HOW TO RUN 1D LSTM:


HOW TO RUN 2D CNN-LSTM:
1. Download the data by running downloading_2D.py
2. Pre-process data by running preprocessing_2D.py, feel free to visualize data by running visualize_2D_data.py
3. Run wave_tranforms.py
4. Run load_and_stack_era5_variables() in cnn_lstm.py to stack the ERA5 data into one dataset
5. Run combine_variables() in cnn_lstm.py to combine ERA5 data with OSISAF data
6. Run main method in cnn_lstm.py

CREDITS:
Duru Huseyni, Kamya Raman, Jinho Lee