# Stock Price Predictor: Implementing Stocks Predictive Model Using Deep Learning
August 2024, [International Journal of Computing Sciences Research](https://stepacademic.net/ijcsr) 8:3147-3156\
[DOI:10.25147/ijcsr.2017.001.1.209](http://dx.doi.org/10.25147/ijcsr.2017.001.1.209)

The Stonks API provides tools for preprocessing stock market data, training machine learning models, and making predictions on future stock prices.

## Overview

*Purpose* – This paper proposes a novel deep neural network model, specifically long short-term memory (LSTM) networks, for predicting stock prices using historical data and financial indicators.

*Method* – LTSM can handle long sequences while capturing temporal dependencies, making it an excellent choice for NLP or time series. The model is trained and tested on the Ayala Corporation (AYALY) stock dataset from 2016 to 2019, using four financial indicators: earnings per share (EPS), EPS growth, price/earnings ratio, and price/earnings-to-growth ratio.

*Results* – The results show that the model achieves high accuracy and outperforms other Deep Neural Network variants as confirmed by assessing its performance using suitable metrics like mean squared error and mean absolute error. It effectively explored and selected relevant financial indicators, implemented data preprocessing techniques, and trained the model using historical data.

*Conclusion* – The project effectively explored and selected relevant financial indicators and trained LSTM models using historical data, and, thus, met its objectives to develop a deep neural network model for stock price prediction.

*Recommendations* – The authors recommend that future researchers continue to explore the integration of a diverse set of financial indicators, employ rigorous comparative analyses, and experiment with different time frames for future predictions to further enhance prediction accuracy.

*Keywords* – Deep Neural Network, Long Short-Term Memory (LSTM) Networks, Machine Learning, Stock Price Prediction, Time Series Forecasting

## Usage

### Loading and Preprocessing Data

#### Import necessary modules
```python
import Stonks
import pandas as pd
```

#### Load stock market data from a CSV file
```python
df = pd.read_csv("dataset.csv", parse_dates=['Date'])
```

#### Preprocess the data for future predictions
```python
data, scaler = Stonks.Processing.preprocess_stock_data(df, future_steps=7)
```

### Building a Prediction Model

#### Load a pre-trained model
```python
model = Stonks.Model.load('lstm')
```

#### Define independent and dependent variables
```python
independent = ['Close_future_1', 'Close_future_2', 'Close_future_3',
               'Close_future_4', 'Close_future_5', 'Close_future_6', 'Close_future_7']
X = data.drop(['Close', 'Date'] + independent, axis=1).values
y = data[independent].values
```

#### Make predictions using the model
```python
y_pred = model.predict(X)
```

### Post-Processing & Visualization

#### Convert predictions back to original scale
```python
inversed_y_pred = [round((i * (scaler.data_max_[31] - scaler.data_min_[31]) + scaler.data_min_[31]), 2) for i in y_pred[50]]
```

#### Generate a stock price prediction graph
```python
import numpy as np
import matplotlib.pyplot as plt

date = range(1, 38)
plt.plot(date, np.concatenate((plot_x_lookback, inversed[50][31:38])), label='Actual')
plt.plot(date[30:37], np.array(inversed_y_pred), label='Predictions')
plt.title("Bank of the Philippine Islands (BPHLY)")
plt.xlabel("Days")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.grid(True)
plt.show()
```

### Performance Evaluation

#### Evaluate prediction accuracy using error metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(inversed_y_pred, inversed[50][31:38])
mae = mean_absolute_error(inversed_y_pred, inversed[50][31:38])

print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
```
