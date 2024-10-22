# Sensor Data Filtering and Time-Series Analysis
This project provides a complete pipeline for fetching, processing, and analyzing time-series sensor data using a custom API. It includes model training for sensor data forecasting, dynamic time warping (DTW) analysis, and RMSE evaluation. The system automates sensor data collection, pre-processes it for machine learning models, and visualizes results with clear, informative plots.

### Features:
- **API Integration:** Fetch sensor data from a custom API for both input and output sensors.
- **Dynamic Time Warping (DTW) Analysis:** Compare time-series data from multiple sensors to find similarities using DTW matrices.
- **Machine Learning Models:** Train LSTM, GRU, LSTM-GRU, and TCN models for sensor data prediction.
- **Error Metrics:** Evaluate model performance using RMSE and save the results to Excel files.
- **Data Visualization:** Automatically generate plots of sensor data and DTW matrices for analysis and comparison.

### Key Components:
- **Data Fetching:** Pulls real-time sensor data via API calls and saves it in structured directories.
- **Model Training & Evaluation:** Trains multiple deep learning models for time-series prediction and evaluates them based on performance.
- **Dynamic Time Warping Analysis:** Processes sensor data, calculates DTW distance matrices, and visualizes the most similar sensors.
- **Visualization & Reporting:** Generates and saves sensor data plots, DTW matrices, and RMSE values in an organized folder structure.

### Setup & Usage:
1. Clone the repository.
2. Follow the instructions in the README.txt file to install dependencies and configure the environment step by step.
3. Define paths for saving models, specify output directories, and adjust the list of sensors in main.py.
4. Run the script to fetch data, train models, evaluate performance, and generate results.
