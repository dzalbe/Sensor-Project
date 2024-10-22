import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from .pyDashboard import pyDashboard
from .apiConfig import ApiConfig


def check_data_integrity(df, step_description, upper_bound, lower_bound):
    if df.empty:
        logging.warning(f"Data is empty after {step_description}")
    else:
        if (df['value'] > upper_bound).any():
            logging.warning(
                f"Data contains values above the expected range after {step_description}")
        if (df['value'] < lower_bound).any():
            logging.warning(
                f"Data contains values below the expected range after {step_description}")


def get_time_range_days(start_timestamp, end_timestamp):
    delta = end_timestamp - start_timestamp
    return delta.days


def fetch_single_sensor_data(sensor_ids, start_timestamp, end_timestamp):
    api_base_url = ApiConfig.base_url
    api_key = ApiConfig.api_key

    all_sensor_data = {}

    for sensor_id in sensor_ids:
        print(f"Fetching data for sensor ID: {sensor_id}")

        history_endpoint = 'v1/measurements/history'
        api_url = f"{api_base_url}/{history_endpoint}?sensor={sensor_id}&from={start_timestamp.strftime('%Y-%m-%dT%H:%M:%S')}&to={end_timestamp.strftime('%Y-%m-%dT%H:%M:%S')}"

        headers = {'accept': 'application/json', 'ApiKey': api_key}
        print(f"Making API call to: {api_url}")

        response = requests.get(api_url, headers=headers)
        print(
            f"Response Status Code: {response.status_code} for Sensor ID: {sensor_id}")

        if response.status_code == 200:
            readings = response.json().get('readings', [])
            if readings:
                df = pd.DataFrame(readings)
                print(
                    f"All available columns for sensor {sensor_id}: {df.columns.tolist()}")
                if 'time' in df.columns and 'value' in df.columns:
                    df = df[df['metric'] == '1']
                    df.reset_index(drop=True, inplace=True)
                    df.index.name = 'idx'
                    all_sensor_data[sensor_id] = df[['time', 'value']].copy()
                else:
                    print(f"Missing required columns for sensor {sensor_id}.")
            else:
                print(f"No readings found for sensor {sensor_id}.")
        else:
            print(
                f"Error fetching data for sensor {sensor_id}: {response.status_code} - {response.text}")

    return all_sensor_data
