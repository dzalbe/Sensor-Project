import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_all_sensor_data(sensor_data, run_folder):
    plt.figure(figsize=(14, 6))

    for sensor_id, data in sensor_data.items():
        plt.plot(pd.to_datetime(data['time']), data['value'],
                 label=f'Sensor {sensor_id}')

    plt.title('Sensor Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, 'sensor_data.png'))
    plt.show()
