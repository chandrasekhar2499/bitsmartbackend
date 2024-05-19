from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

app = Flask(__name__)
CORS(app)  # Apply CORS to the Flask app

# Load the trained model and scaler
model = load_model('btc_price_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fetch and preprocess data
btc_data = yf.download('BTC-USD', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
btc_data.index = pd.to_datetime(btc_data.index)
btc_data['Rolling_Mean_7'] = btc_data['Close'].rolling(window=7).mean().fillna(method='bfill')
btc_data['Rolling_Std_7'] = btc_data['Close'].rolling(window=7).std().fillna(method='bfill')
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean_7', 'Rolling_Std_7']
btc_data[features_to_scale] = scaler.transform(btc_data[features_to_scale])

def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i][features_to_scale].values)
        y.append(data.iloc[i][['High', 'Low', 'Close']].values)
    return np.array(X), np.array(y)

X, y = create_sequences(btc_data)

@app.route('/predict', methods=['GET'])
def predict():
    start_date = request.args.get('start_date', default=datetime.now().strftime('%Y-%m-%d'))
    
    if pd.to_datetime(start_date) not in btc_data.index:
        start_date = btc_data.index[-1]
    
    idx = btc_data.index.get_loc(start_date)
    last_sequences = btc_data.iloc[idx-7:idx][features_to_scale].values.reshape(1, 7, len(features_to_scale))
    predictions = model.predict(last_sequences)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], len(features_to_scale) - 3)))))[:, :3]
    
    highest_price = float(np.max(predictions[:, 0]))
    lowest_price = float(np.min(predictions[:, 1]))
    avg_closing_price = float(np.mean(predictions[:, 2]))
    
    return jsonify({
        'highest_price': highest_price,
        'lowest_price': lowest_price,
        'avg_closing_price': avg_closing_price
    })

@app.route('/trade', methods=['GET'])
def trade():
    start_date = request.args.get('start_date', default=datetime.now().strftime('%Y-%m-%d'))
    
    if pd.to_datetime(start_date) not in btc_data.index:
        start_date = btc_data.index[-1]
    
    idx = btc_data.index.get_loc(start_date)
    initial_sequences = btc_data.iloc[idx-7:idx][features_to_scale].values.reshape(1, 7, len(features_to_scale))
    
    # Generate 7-day predictions
    predictions = []
    current_sequence = initial_sequences
    for _ in range(7):
        prediction = model.predict(current_sequence)
        predictions.append(prediction[0])
        new_row = np.hstack((prediction, np.zeros((1, len(features_to_scale) - 3))))
        current_sequence = np.hstack((current_sequence[:, 1:], new_row[:, np.newaxis]))

    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], len(features_to_scale) - 3)))))[:, :3]

    initial_cash = 100000
    initial_open_price = btc_data.iloc[idx]['Open']
    bitcoins = initial_cash / initial_open_price
    
    best_trade = {
        'sell_date': None,
        'load_date': None,
        'final_cash': initial_cash
    }

    for i in range(7):
        sell_price = predictions[i, 2]  # Open price for selling
        cash_after_sell = bitcoins * sell_price
        
        for j in range(i + 1, 7):
            load_price = predictions[j, 2]  # Open price for loading
            bitcoins_after_load = cash_after_sell / load_price
            final_cash = bitcoins_after_load * predictions[6, 2]  # Closing price on the 7th day
            
            if final_cash > best_trade['final_cash']:
                best_trade = {
                    'sell_date': (pd.to_datetime(start_date) + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                    'load_date': (pd.to_datetime(start_date) + pd.Timedelta(days=j)).strftime('%Y-%m-%d'),
                    'final_cash': final_cash
                }

    if best_trade['sell_date'] is None:
        best_trade['sell_date'] = 'NA'
    if best_trade['load_date'] is None:
        best_trade['load_date'] = 'NA'

    return jsonify(best_trade)

if __name__ == '__main__':
    app.run(debug=True)
