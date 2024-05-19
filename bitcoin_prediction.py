import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from datetime import datetime

# Fetch data
btc_data = yf.download('BTC-USD', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))

# Ensure the index is datetime for better handling
btc_data.index = pd.to_datetime(btc_data.index)

# Feature engineering
btc_data['Rolling_Mean_7'] = btc_data['Close'].rolling(window=7).mean().fillna(method='bfill')
btc_data['Rolling_Std_7'] = btc_data['Close'].rolling(window=7).std().fillna(method='bfill')

# Normalize data
scaler = MinMaxScaler()
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean_7', 'Rolling_Std_7']
btc_data[features_to_scale] = scaler.fit_transform(btc_data[features_to_scale])

# Prepare data for LSTM input
def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i][features_to_scale].values)
        y.append(data.iloc[i][['High', 'Low', 'Close']].values)
    return np.array(X), np.array(y)

X, y = create_sequences(btc_data)

# Define the LSTM model
model = Sequential([
    LSTM(50, input_shape=(7, len(features_to_scale)), return_sequences=True),
    LSTM(50),
    Dense(3)  # Outputs: High, Low, Close
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Function to predict future prices
def predict_and_trade(start_date):
    # Check if the start date is within the dataset
    if pd.to_datetime(start_date) not in btc_data.index:
        print(f"Start date {start_date} is beyond the available data. Using the last available data for predictions.")
        start_date = btc_data.index[-1]

    idx = btc_data.index.get_loc(start_date)
    last_sequences = btc_data.iloc[idx-7:idx][features_to_scale].values.reshape(1, 7, len(features_to_scale))
    predictions = model.predict(last_sequences)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], len(features_to_scale) - 3)))))[:, :3]

    # Trading strategy
    initial_cash = 100000  # $100,000
    buy_price = predictions[0, 2]  # Use the predicted close price of the first day to buy
    bitcoins = initial_cash / buy_price
    sell_price = np.max(predictions[:, 0])  # Sell at the predicted highest price within the next 7 days
    final_cash = bitcoins * sell_price

    return predictions, final_cash

# Example prediction and trading
start_date = '2024-05-06'
predicted_prices, final_cash = predict_and_trade(start_date)
print(f"Predicted High, Low, and Closing Prices for the next 7 days from {start_date} are:", predicted_prices)
print(f"Final cash after trading based on predictions: ${final_cash:,.2f}")
# Save the trained model and scaler
model.save('btc_price_prediction_model.h5')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

