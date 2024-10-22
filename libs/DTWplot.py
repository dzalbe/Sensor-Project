import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.metrics import dtw
from itertools import product
import seaborn as sns


def process_sensor_data(sensor_data):
    # Combine all sensor data into a single DataFrame
    all_sensor_data = pd.concat(sensor_data.values(
    ), keys=sensor_data.keys(), names=['sensor_id', 'idx'])

    # Ensure the 'idx' and 'value' columns exist
    if 'value' in all_sensor_data.columns:
        all_sensor_data.reset_index(inplace=True)
    else:
        raise KeyError("Expected columns 'idx' and 'value' not found in data.")

    return all_sensor_data


def calculate_dtw_matrix(df_sensordata):
    selected_sensor_ids = df_sensordata['sensor_id'].unique()
    num_sensors = len(selected_sensor_ids)
    distance_matrix = np.zeros((num_sensors, num_sensors))

    for i, sensor1_id in enumerate(selected_sensor_ids):
        for j, sensor2_id in enumerate(selected_sensor_ids):
            sensor1_data = df_sensordata[df_sensordata['sensor_id']
                                         == sensor1_id]['value'].values
            sensor2_data = df_sensordata[df_sensordata['sensor_id']
                                         == sensor2_id]['value'].values
            if len(sensor1_data) == 0 or len(sensor2_data) == 0:
                continue
            distance_matrix[i][j] = dtw(sensor1_data, sensor2_data)

    return distance_matrix, selected_sensor_ids


def plot_dtw_matrix(distance_matrix, selected_sensor_ids, run_folder):
    plt.figure(figsize=(7, 7))
    plt.imshow(distance_matrix, cmap='viridis',
               vmin=0, vmax=np.max(distance_matrix))
    plt.colorbar()
    plt.xticks(range(len(selected_sensor_ids)),
               selected_sensor_ids, rotation=90)
    plt.yticks(range(len(selected_sensor_ids)), selected_sensor_ids)
    plt.title('Sensor Distance Matrix (DTW)')
    plt.savefig(os.path.join(run_folder, 'DTW_matrix.png'))
    plt.show()


def save_dtw_distances_to_excel(distance_matrix, selected_sensor_ids, run_folder):
    df_distances = pd.DataFrame(
        distance_matrix, index=selected_sensor_ids, columns=selected_sensor_ids)
    excel_path = os.path.join(run_folder, 'DTW_distances.xlsx')
    df_distances.to_excel(excel_path, engine='openpyxl')
    print(f"DTW distances saved to {excel_path}")


def plot_dtw_matrix_io(distance_matrix, input_sensors, output_sensors, run_folder):
    # Ensure that the distance matrix has the correct shape
    if distance_matrix.shape != (len(input_sensors), len(output_sensors)):
        raise ValueError(
            "Distance matrix shape does not match the length of input and output sensors lists")

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(distance_matrix, xticklabels=output_sensors,
                     yticklabels=input_sensors, cmap='viridis', annot=True, fmt=".2f")
    ax.set_title('DTW Distance Matrix (Input Sensors vs Output Sensors)')
    ax.set_xlabel('Output Sensors')
    ax.set_ylabel('Input Sensors')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, 'DTW_matrix_io.png'))
    plt.close()


def print_most_similar_sensors(distance_matrix, input_sensors, output_sensors):
    print("\nMost similar input sensor for each output sensor based on DTW:")
    print(f"Output Sensor | Input Sensor | DTW Value")
    for j, output_sensor_id in enumerate(output_sensors):
        min_dtw_value = np.min(distance_matrix[:, j])
        most_similar_input_idx = np.argmin(distance_matrix[:, j])
        most_similar_input_id = input_sensors[most_similar_input_idx]
        print(f"{output_sensor_id} | {most_similar_input_id} | {min_dtw_value:.4f}")
