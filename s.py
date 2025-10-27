# s.py
from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os

app = Flask(__name__)

# Utility: prepare sequences for LSTM
def create_sequences(values, seq_length):
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length])
    return np.array(X), np.array(y)

# Build a small LSTM model
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build a small ANN (Dense) model
def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

@app.route('/')
def home():
    return jsonify({"message": "Stock prediction service (LSTM + ANN). Use /predict?ticker=MSFT&period=1y"})

@app.route('/predict', methods=['GET'])
def predict():
    """
    Endpoint:
    /predict?ticker=MSFT&period=1y&seq=10&epochs=2
    - ticker: ticker symbol (default: AAPL)
    - period: e.g. 6mo, 1y, 2y (default: 1y)
    - seq: sequence length for LSTM (default: 10)
    - epochs: training epochs (default: 2)
    """
    # Read query params
    ticker = request.args.get('ticker', 'AAPL').upper()
    period = request.args.get('period', '1y')
    seq_length = int(request.args.get('seq', 10))
    epochs = int(request.args.get('epochs', 2))

    # Safety limits (to avoid huge training on serverless)
    if seq_length < 1:
        seq_length = 10
    if epochs < 1:
        epochs = 1
    if epochs > 5:
        # cap epochs to 5 for serverless safety
        epochs = 5

    try:
        # Fetch historical data
        df = yf.download(ticker, period=period, progress=False)
        if df.empty or 'Close' not in df.columns:
            return jsonify({"error": "No data found for ticker/period"}), 400

        # Use only Close price
        close = df['Close'].values.reshape(-1, 1)

        # Optionally downsample if too large
        max_len = 1000
        if len(close) > max_len:
            close = close[-max_len:]

        # Scale
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        # Prepare sequences
        X, y = create_sequences(scaled.flatten(), seq_length)
        if len(X) < 5:
            return jsonify({"error": "Not enough data after sequence splitting. Use a longer period."}), 400

        # Use last portion for training only (we're training on all available small dataset)
        # Reshape for LSTM: (samples, seq_length, 1)
        X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train LSTM
        lstm = build_lstm((X_lstm.shape[1], X_lstm.shape[2]))
        # small training for quickness - verbose 0 to reduce log noise
        lstm.fit(X_lstm, y, epochs=epochs, batch_size=8, verbose=0)

        # Predict next (use last seq from data)
        last_seq = scaled.flatten()[-seq_length:]
        lstm_input = last_seq.reshape((1, seq_length, 1))
        lstm_pred_scaled = lstm.predict(lstm_input, verbose=0)[0][0]
        lstm_pred = float(scaler.inverse_transform(np.array([[lstm_pred_scaled]]))[0][0])

        # Build and train ANN on flattened features (each sample is seq_length features)
        X_ann = X  # shape (samples, seq_length)
        ann = build_ann(X_ann.shape[1])
        ann.fit(X_ann, y, epochs=epochs, batch_size=8, verbose=0)
        ann_input = last_seq.reshape((1, seq_length))
        ann_pred_scaled = ann.predict(ann_input, verbose=0)[0][0]
        ann_pred = float(scaler.inverse_transform(np.array([[ann_pred_scaled]]))[0][0])

        # Also provide last actual close for reference
        last_actual = float(close[-1][0])

        return jsonify({
            "ticker": ticker,
            "period": period,
            "seq_length": seq_length,
            "epochs": epochs,
            "last_actual_close": last_actual,
            "predicted_next_close": {
                "lstm": round(lstm_pred, 4),
                "ann": round(ann_pred, 4),
            },
            "note": "Models are small and trained briefly for demo. For production, pre-train offline and serve saved models."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local dev server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
