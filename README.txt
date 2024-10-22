To successfully run the project on your computer, some changes need to be applied in main.py:

1. Define your "output_directory" on line 65 – this folder will store all the project's results.
2. Choose the time period for which to obtain sensor data (lines 68-70) – you can define days, weeks or months.
3. If necessary, modify the lists of input and output sensors – lines 14 and 16.
4. Adjust the "MODEL_DIR" path on line 34 according to the chosen "output_directory."
5. In GetSingleSensor.py, you can change the measurement unit on line 53:

1. 1 = temperature
2. 2 = humidity
3. 3 = CO2
4. 4 = atmospheric pressure
To modify the configuration of model network architectures, changes need to be made in testModels.py on lines 21-28.

Program Description:
main.py – the main execution file that coordinates the entire project's operation.
apiConfig.py – contains Base URL and API Key information.
GetSingleSensor.py – retrieves the defined sensor data.
calculateTimeRange.py – helper function that calculates the start and end timestamps based on the time period defined in main.py.
dataFetching.py – organizes the retrieval and storage of sensor data in the specified folders.
DTWplot.py – performs "Dynamic Time Warping" calculations between input and output sensors, visualizing the results in DTW matrices.
plotSensorData.py – generates and saves graphs for all retrieved sensor data.
pyDashboard.py – manages communication with the API, allowing the retrieval of necessary sensor information.
sensorInfoUpdate.py – updates the "Taken_info.xlsx" file, which records information about the retrieved sensor data.
testModels.py – defines, trains, and evaluates four machine learning models for sensor value prediction, where RMSE evaluations are also calculated and stored.