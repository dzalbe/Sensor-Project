import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl import Workbook
from datetime import datetime


def create_initial_excel(file_path, sensor_ids):
    # Create a new DataFrame with sensor IDs as columns
    df = pd.DataFrame(columns=sensor_ids)
    df.to_excel(file_path, index=False)


def update_excel_with_sensor_info(sensor_ids, days_fetched, output_directory, input_sensors, output_sensors, start_timestamp, end_timestamp):
    file_path = os.path.join(output_directory, "Taken_info.xlsx")
    date_run = datetime.now().strftime("%Y-%m-%d")

    # Check if file exists and create it if it doesn't
    if not os.path.exists(file_path):
        create_initial_excel(file_path, sensor_ids)

    # Load the existing Excel file
    try:
        book = load_workbook(file_path)
    except Exception as e:
        print(f"Error loading workbook: {e}")
        return

    # Read the existing data
    try:
        existing_df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Create a new row with the date and days fetched
    new_row = {sensor_id: None for sensor_id in sensor_ids}
    for sensor_id in input_sensors:
        new_row[sensor_id] = f"IN: {date_run} - {days_fetched} days ({start_timestamp.date()} to {end_timestamp.date()})"
    for sensor_id in output_sensors:
        new_row[sensor_id] = f"OUT: {date_run} - {days_fetched} days ({start_timestamp.date()} to {end_timestamp.date()})"

    # Convert the new row to a DataFrame
    new_row_df = pd.DataFrame([new_row])

    # Append the new row to the DataFrame
    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)

    # Save the updated DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        updated_df.to_excel(writer, index=False, sheet_name='Sheet1')
