import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os


class Config:
    hidden_layer_size = 20
    sequence_length = 50
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    number_of_sensors = 10  # Update this to reflect the actual number of sensors
    input_size = number_of_sensors


class MultiSensorLSTM(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.lstm = nn.LSTM(config.input_size,
                            config.hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(config.hidden_layer_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x


class MultiSensorGRU(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.gru = nn.GRU(config.input_size,
                          config.hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(config.hidden_layer_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x[:, -1, :])
        return x


class MultiSensorLSTM_GRU(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.lstm = nn.LSTM(config.input_size,
                            config.hidden_layer_size, batch_first=True)
        self.gru = nn.GRU(config.hidden_layer_size,
                          config.hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(config.hidden_layer_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x = self.linear(x[:, -1, :])
        return x


class IntegratedTCN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        layers = []
        num_channels = [config.hidden_layer_size] * 4  # Adjustable depth
        kernel_size = 2
        dropout = 0.2
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = config.input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
                                    padding=dilation_size * (kernel_size - 1), dilation=dilation_size))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if i < len(num_channels) - 1:
                layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Change to (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.linear(x)
        return x


def load_data(input_paths, output_paths, config):
    # Load and preprocess data
    input_data = []
    max_length = 0

    # Fetch input data and find the maximum length
    for file_path in input_paths:
        sensor_data = pd.read_csv(file_path, delimiter=';')
        print(f"Columns in {file_path}: {sensor_data.columns.tolist()}")
        if 'value' in sensor_data.columns:
            sensor_values = sensor_data['value'].values
            input_data.append(sensor_values)
            max_length = max(max_length, len(sensor_values))
        else:
            print(f"Skipping file {file_path} as 'value' column is not found")

    # Ensure there's data to process
    if not input_data:
        raise ValueError("No valid input data found")

    # Pad input data sequences to the maximum length
    padded_input_data = []
    for data in input_data:
        padded_data = np.pad(data, (0, max_length - len(data)),
                             'constant', constant_values=np.nan)
        padded_input_data.append(padded_data)

    # Convert to numpy array and handle missing values
    input_data = np.array(padded_input_data).T
    input_data = np.nan_to_num(input_data)

    # Normalize input data
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)

    # Load and preprocess output data
    output_data = []
    max_length = 0

    # Fetch output data and find the maximum length
    for file_path in output_paths:
        sensor_data = pd.read_csv(file_path, delimiter=';')
        print(f"Columns in {file_path}: {sensor_data.columns.tolist()}")
        if 'value' in sensor_data.columns:
            sensor_values = sensor_data['value'].values
            output_data.append(sensor_values)
            max_length = max(max_length, len(sensor_values))
        else:
            print(f"Skipping file {file_path} as 'value' column is not found")

    # Ensure there's data to process
    if not output_data:
        raise ValueError("No valid output data found")

    # Pad output data sequences to the maximum length
    padded_output_data = []
    for data in output_data:
        padded_data = np.pad(data, (0, max_length - len(data)),
                             'constant', constant_values=np.nan)
        padded_output_data.append(padded_data)

    # Convert to numpy array and handle missing values
    output_data = np.array(padded_output_data).T
    output_data = np.nan_to_num(output_data)

    # Normalize output data
    output_data = scaler.fit_transform(output_data)

    # Create sequences
    X, y = [], []
    for i in range(len(input_data) - config.sequence_length):
        if i + config.sequence_length < len(output_data):
            X.append(input_data[i:i + config.sequence_length])
            y.append(output_data[i + config.sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


def train_model(model, X_train, y_train, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()

    for epoch in range(config.epochs):
        for i in range(0, len(X_train), config.batch_size):
            X_batch = torch.tensor(
                X_train[i:i + config.batch_size], dtype=torch.float32)
            y_batch = torch.tensor(
                y_train[i:i + config.batch_size], dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{config.epochs}], Loss: {loss.item():.4f}')


def save_rmse_to_excel(rmse_values, input_sensors, rmse_excel_path):
    data = []
    for sensor_id, model_rmses in rmse_values.items():
        for model_name, rmse in model_rmses.items():
            data.append({
                'Output Sensor ID': sensor_id,
                'Model': model_name,
                'RMSE': rmse,
                'Input Sensors': ', '.join(input_sensors)
            })
    df = pd.DataFrame(data)
    df.to_excel(rmse_excel_path, index=False, engine='openpyxl')


def evaluate_model(model, X_test, y_test, scaler, model_name, output_paths):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        predictions = model(X_test).numpy()
        y_test = y_test.numpy()

        # Reshape to 2D array
        y_test = y_test.reshape(-1, y_test.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])

        # Inverse transform to original scale
        true_values_original = scaler.inverse_transform(y_test)
        predictions_original = scaler.inverse_transform(predictions)

        rmses = {}

        for i in range(true_values_original.shape[1]):
            sensor_id = os.path.basename(output_paths[i]).split('.')[0]
            rmse = sqrt(mean_squared_error(
                true_values_original[:, i], predictions_original[:, i]))
            rmses[sensor_id] = rmse
            print(
                f'Root Mean Squared Error ({model_name}, Sensor {sensor_id}): {rmse:.4f}')

        return true_values_original, predictions_original, rmses


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path} \n')


def plot_all_predictions(true_values, predictions_dict, sensor_id, output_directory):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values')
    for model_name, predictions in predictions_dict.items():
        plt.plot(predictions, label=f'Predictions {model_name}')
    plt.legend()
    plt.grid(True)
    plt.title(f'Predictions for Sensor {sensor_id}')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.savefig(os.path.join(output_directory,
                f'{sensor_id}_1-full.png'))
    plt.close()

    # Plot first 200 predictions
    plt.figure(figsize=(10, 5))
    plt.plot(true_values[:200], label='True Values (First 200)')
    for model_name, predictions in predictions_dict.items():
        plt.plot(predictions[:200],
                 label=f'Predictions {model_name} (First 200)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Predictions for Sensor {sensor_id} (First 200)')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.savefig(os.path.join(output_directory,
                f'{sensor_id}_2-first_200.png'))
    plt.close()

    # Plot first 50 predictions
    plt.figure(figsize=(10, 5))
    plt.plot(true_values[:50], label='True Values (First 50)')
    for model_name, predictions in predictions_dict.items():
        plt.plot(predictions[:50],
                 label=f'Predictions {model_name} (First 50)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Predictions for Sensor {sensor_id} (First 50)')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.savefig(os.path.join(output_directory,
                f'{sensor_id}_3-first_50.png'))
    plt.close()


def main(config, input_paths, output_paths, output_directory, model_dir):

    # Define paths to save models using model_dir from main.py
    GRU_MODEL_PATH = os.path.join(model_dir, 'GRU.pth')
    LSTM_MODEL_PATH = os.path.join(model_dir, 'LSTM.pth')
    LSTM_GRU_MODEL_PATH = os.path.join(model_dir, 'LSTM-GRU.pth')
    TCN_MODEL_PATH = os.path.join(model_dir, 'TCN.pth')

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_data(
        input_paths, output_paths, config)

    output_size = y_train.shape[-1]

    # Initialize models
    gru_model = MultiSensorGRU(config, output_size)
    lstm_model = MultiSensorLSTM(config, output_size)
    lstm_gru_model = MultiSensorLSTM_GRU(config, output_size)
    tcn_model = IntegratedTCN(config, output_size)

    # Train and evaluate GRU model
    print("Training GRU model...")
    train_model(gru_model, X_train, y_train, config)
    print("Evaluating GRU model...")
    true_values, predictions_gru, rmses_gru = evaluate_model(
        gru_model, X_test, y_test, scaler, "GRU", output_paths)
    save_model(gru_model, GRU_MODEL_PATH)

    # Train and evaluate LSTM model
    print("Training LSTM model...")
    train_model(lstm_model, X_train, y_train, config)
    print("Evaluating LSTM model...")
    _, predictions_lstm, rmses_lstm = evaluate_model(
        lstm_model, X_test, y_test, scaler, "LSTM", output_paths)
    save_model(lstm_model, LSTM_MODEL_PATH)

    # Train and evaluate LSTM-GRU model
    print("Training LSTM-GRU model...")
    train_model(lstm_gru_model, X_train, y_train, config)
    print("Evaluating LSTM-GRU model...")
    _, predictions_lstm_gru, rmses_lstm_gru = evaluate_model(
        lstm_gru_model, X_test, y_test, scaler, "LSTM-GRU", output_paths)
    save_model(lstm_gru_model, LSTM_GRU_MODEL_PATH)

    # Train and evaluate TCN model
    print("Training TCN model...")
    train_model(tcn_model, X_train, y_train, config)
    print("Evaluating TCN model...")
    _, predictions_tcn, rmses_tcn = evaluate_model(
        tcn_model, X_test, y_test, scaler, "TCN", output_paths)
    save_model(tcn_model, TCN_MODEL_PATH)

    # Log RMSE values for each output sensor
    rmse_values = {}
    for i, sensor_path in enumerate(output_paths):
        sensor_id = os.path.basename(sensor_path).split('.')[0]
        rmse_values[sensor_id] = {
            'GRU': rmses_gru[sensor_id],
            'LSTM': rmses_lstm[sensor_id],
            'LSTM-GRU': rmses_lstm_gru[sensor_id],
            'TCN': rmses_tcn[sensor_id]
        }
        print(f'RMSE for GRU - Sensor {sensor_id}: {rmses_gru[sensor_id]:.4f}')
        print(
            f'RMSE for LSTM - Sensor {sensor_id}: {rmses_lstm[sensor_id]:.4f}')
        print(
            f'RMSE for LSTM-GRU - Sensor {sensor_id}: {rmses_lstm_gru[sensor_id]:.4f}')
        print(f'RMSE for TCN - Sensor {sensor_id}: {rmses_tcn[sensor_id]:.4f}')

    # Plot all predictions for each output sensor
    for i, sensor_path in enumerate(output_paths):
        sensor_id = os.path.basename(sensor_path).split('.')[0]
        predictions_dict = {
            'GRU': predictions_gru[:, i],
            'LSTM': predictions_lstm[:, i],
            'LSTM-GRU': predictions_lstm_gru[:, i],
            'TCN': predictions_tcn[:, i]
        }
        plot_all_predictions(
            true_values[:, i], predictions_dict, sensor_id, output_directory)

    return rmse_values


if __name__ == "__main__":
    config = Config()
    input_sensors = []  # Define input sensors here if running standalone
    output_sensors = []  # Define output sensors here if running standalone
    main(config, input_sensors, output_sensors)
