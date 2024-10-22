import os
import pandas as pd
from .GetSingleSensor import fetch_single_sensor_data
from .apiConfig import ApiConfig


def fetch_sensor_data(sensor_ids, input_sensors, output_sensors, start_timestamp, end_timestamp, output_directory):
    all_sensor_data = fetch_single_sensor_data(
        sensor_ids, start_timestamp, end_timestamp)

    input_sensors_dir = os.path.join(output_directory, 'input_sensors')
    output_sensors_dir = os.path.join(output_directory, 'output_sensors')

    # Ensure directories exist
    os.makedirs(input_sensors_dir, exist_ok=True)
    os.makedirs(output_sensors_dir, exist_ok=True)

    for sensor_id, df in all_sensor_data.items():
        if 'value' not in df.columns:
            continue

        if sensor_id in input_sensors:
            subfolder = input_sensors_dir
        else:
            subfolder = output_sensors_dir

        # Save the DataFrame to CSV
        new_csv_file_path = os.path.join(subfolder, f'{sensor_id}.csv')
        df[['value']].to_csv(new_csv_file_path, sep=';', index=True)

    return all_sensor_data
