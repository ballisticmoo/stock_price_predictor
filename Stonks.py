import pandas as pd
import tensorflow as tf
from  sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path

class Processing:

    def preprocess_stock_data(df, target_column='Close', look_back=30, future_steps=1):
        """
        Preprocesses stock data for time series forecasting.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing stock data with a 'Date' column and the specified target column.
        - target_column (str): Name of the target column to be used for forecasting (default is 'Close').
        - look_back (int): Number of past time steps to include as features (default is 30).
        - future_steps (int): Number of future time steps to include as target values (default is 1).

        Returns:
        - pd.DataFrame: Processed DataFrame with normalized values, lag features, and future features.
        """

        # Sort the DataFrame by date
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        # Use only the relevant columns (Date and target_column)
        df = df[['Date', target_column]]

        # Create a new DataFrame with shifted target values
        for i in range(1, look_back + 1):
            df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)

        for i in range(1, future_steps+1):
            df[f"{target_column}_future_{i}"] = df[target_column].shift(1-i)

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Normalize the data using Min-Max scaling
        scaler = MinMaxScaler()
        df[[target_column] + 
        [f'{target_column}_lag_{i}' for i in range(1, look_back + 1)] + 
        [f"{target_column}_future_{i}" for i in range(1, future_steps+1)]] = scaler.fit_transform(
            df[[target_column] + [f'{target_column}_lag_{i}' for i in range(1, look_back + 1)] + 
            [f"{target_column}_future_{i}" for i in range(1, future_steps+1)]]
        )

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df
    
class Model:

    def download(model):
        """
        Downloads a pre-trained stock price prediction model from the specified list of models.

        Parameters:
        - model (str): Name of the pre-trained model to download. Choose from: 'eps', 'epsgrowth', 'lstm', 'pe', 'peg'.

        Returns:
        - tf.keras.Model: Loaded pre-trained model for stock price prediction.

        Raises:
        - ValueError: If the specified model is not in the list.
        - RuntimeError: If the download fails or the response status code is not 200.
        """

        model_urls = {'eps': "https://github.com/ballisticmoo/stock_price_predictor/raw/main/eps.h5",
                     'epsgrowth': "https://github.com/ballisticmoo/stock_price_predictor/raw/main/epsgrowth.h5",
                     'lstm': "https://github.com/ballisticmoo/stock_price_predictor/raw/main/lstm.h5",
                     'pe': "https://github.com/ballisticmoo/stock_price_predictor/raw/main/pe.h5",
                     'peg': "https://github.com/ballisticmoo/stock_price_predictor/raw/main/peg.h5",}
        
        if model not in model_urls:
            raise ValueError(f"Model '{model}' not found in the list.")
        
        model_url = model_urls[model]

            # Download the model
        model_filename = f"{model}.h5"
        download_path = Path(model_filename)

        response = requests.get(model_url)
        if response.status_code == 200:
            with open(download_path, 'wb') as file:
                file.write(response.content)
        else:
            raise RuntimeError(f"Failed to download the model. Status code: {response.status_code}")

        # Load the model
        loaded_model = tf.keras.models.load_model(download_path)

        return loaded_model
    

    def load_architecture(architecture):
        """
        Creates and compiles a neural network model based on the specified architecture.

        Parameters:
        - architecture (str): Name of the neural network architecture to load. Choose from: 'LSTM1', 'LSTM-FI'.

        Returns:
        - tf.keras.Model: Compiled neural network model.

        Raises:
        - ValueError: If the specified architecture is not in the list.
        """

        architectures = {'LSTM1': [LSTM(units=50, return_sequences=True, input_shape=(30, 1)),
                                   LSTM(units=50, return_sequences=True),
                                   LSTM(units=50),
                                   Dropout(0.2),
                                   Dense(units=7)],
                        'LSTM-FI': [LSTM(units=50, return_sequences=True, input_shape=(31, 1)),
                                   LSTM(units=50, return_sequences=True),
                                   LSTM(units=50),
                                   Dropout(0.2),
                                   Dense(units=7)]}
        
        if architecture not in architectures:
            raise ValueError(f"Model architecture '{architecture}' not found in the list.")
        
        model = Sequential()
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model
    
class Evaluate:

    def calculate_errors(y_test, y_pred):
        """
        Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE) between actual and predicted values.

        Parameters:
        - y_test (array-like): Ground truth values.
        - y_pred (array-like): Predicted values.

        Returns:
        - dict: Dictionary containing calculated errors.
        - 'MAE' (float): Mean Absolute Error.
        - 'MSE' (float): Mean Squared Error.
        """

        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)

        return {'MAE': MAE, 'MSE': MSE}
    
    def plot_predictions(y_test, y_pred):
        """
        Plot actual and predicted stock prices over time.

        Parameters:
        - y_test (array-like): Ground truth values.
        - y_pred (array-like): Predicted values.

        Returns:
        - None: Displays a plot of actual and predicted stock prices.
        """

        plt.plot(y_test, color='blue')
        plt.plot(y_pred, color='orange')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend(["actual", "predictions"], loc ="lower right")
        plt.show()

    

