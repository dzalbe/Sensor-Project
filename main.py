from datetime import datetime
from libs.calculateTimeRange import calculate_time_range
from libs.apiConfig import ApiConfig
from libs.dataFetching import fetch_sensor_data
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from libs.sensorInfoUpdate import update_excel_with_sensor_info
from libs.testModels import main as run_model, Config, save_rmse_to_excel
from libs.DTWplot import process_sensor_data, calculate_dtw_matrix, plot_dtw_matrix, save_dtw_distances_to_excel, plot_dtw_matrix_io, print_most_similar_sensors
from libs.plotSensorData import plot_all_sensor_data

input_sensors = ["004"]

output_sensors = ["229", "515", "132",
                  "297", "498", "018", "133", "170"]


sensor_ids = [
    "321", "147", "276", "282", "130", "261", "146", "691",
    "625", "638", "654", "143", "986", "280", "001", "218",
    "616", "265", "655", "256", "184", "125", "023", "105",
]

print("Calling API URL:", ApiConfig.base_url)

# Define paths to save your trained models
MODEL_DIR = "C:\\Users\\MyComputer\\Projects\\Sensor filtering\\libs\\models"
GRU_MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM.pth')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'GRU.pth')
LSTM_GRU_MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM-GRU.pth')
TCN_MODEL_PATH = os.path.join(MODEL_DIR, 'TCN.pth')


def get_next_run_folder(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_folders = [f for f in os.listdir(
        base_path) if os.path.isdir(os.path.join(base_path, f))]
    if not existing_folders:
        return os.path.join(base_path, '1')
    else:
        next_folder_num = max(int(f)
                              for f in existing_folders if f.isdigit()) + 1
        return os.path.join(base_path, str(next_folder_num))


def main():
    output_directory = r'C:\\Users\\MyComputer\\Projects\\Sensor filtering\\output'

    # Define the time period for which you want to get sensor data
    days = 0
    weeks = 0
    months = 1

    # Calculate start and end timestamps
    start_timestamp, end_timestamp = calculate_time_range(days, weeks, months)

    # Calculate the total number of days fetched
    days_fetched_folder_name = f"{start_timestamp.strftime('%Y%m%d')}_{end_timestamp.strftime('%Y%m%d')}"
    days_fetched_folder = os.path.join(
        output_directory, days_fetched_folder_name)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    run_folder = get_next_run_folder(days_fetched_folder)
    os.makedirs(run_folder, exist_ok=True)

    # Create input and output sensors subfolders
    input_sensors_folder = os.path.join(run_folder, "input_sensors")
    output_sensors_folder = os.path.join(run_folder, "output_sensors")
    os.makedirs(input_sensors_folder, exist_ok=True)
    os.makedirs(output_sensors_folder, exist_ok=True)

    # Fetch and save sensor data for both input and output sensors
    all_sensors = input_sensors + output_sensors
    print(f"Fetching data for sensors: {all_sensors}")
    all_sensor_data = fetch_sensor_data(
        all_sensors, input_sensors, output_sensors, start_timestamp, end_timestamp, run_folder)

    # Update the Excel file with the sensor information
    update_excel_with_sensor_info(sensor_ids, days_fetched_folder_name, output_directory,
                                  input_sensors, output_sensors, start_timestamp, end_timestamp)

    # Dynamically set the number of sensors in the config
    config = Config()
    config.number_of_sensors = len(input_sensors)
    config.input_size = len(input_sensors)  # Set the correct input size

    # Get the file paths
    input_paths = [os.path.join(
        input_sensors_folder, f'{sensor_id}.csv') for sensor_id in input_sensors]
    output_paths = [os.path.join(
        output_sensors_folder, f'{sensor_id}.csv') for sensor_id in output_sensors]

    # Run the model training and evaluation
    rmse_values = run_model(config, input_paths,
                            output_paths, run_folder, MODEL_DIR)

    # Define the path to save the RMSE Excel file
    rmse_excel_path = os.path.join(run_folder, 'Sensors-RMSE_values.xlsx')

    # Save RMSE values to the Excel file
    save_rmse_to_excel(rmse_values, input_sensors, rmse_excel_path)

    ###################################### PROCESSING AND PLOTTING DTW MATRIXES #######################################

    # Process and plot DTW matrix
    print("Processing and plotting DTW matrix")
    df_sensordata = process_sensor_data(all_sensor_data)
    distance_matrix, selected_sensor_ids = calculate_dtw_matrix(df_sensordata)
    plot_dtw_matrix(distance_matrix, selected_sensor_ids, run_folder)
    save_dtw_distances_to_excel(
        distance_matrix, selected_sensor_ids, run_folder)

    # Plot DTW matrix for input and output sensors
    input_output_distance_matrix = distance_matrix[:len(
        input_sensors), len(input_sensors):]
    plot_dtw_matrix_io(input_output_distance_matrix,
                       input_sensors, output_sensors, run_folder)

    # Print the most similar sensors based on DTW distance
    print_most_similar_sensors(
        input_output_distance_matrix, input_sensors, output_sensors)

    ############################################ PLOTTING SENSOR DATA ############################################
    print("Plotting all fetched sensor data")
    plot_all_sensor_data(all_sensor_data, run_folder)

    return run_folder


if __name__ == "__main__":
    main()
