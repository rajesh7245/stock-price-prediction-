# Install required packages if not already
# pip install pandas numpy matplotlib tensorflow scikit-learn yfinance

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------------
# 1️⃣ Get Stock Data
# -------------------------------
ticker = 'AAPL'  # Example: Apple
data = yf.download(ticker, start='2018-01-01', end='2025-01-01')
data = data[['Close']]  # We use only the closing price

# -------------------------------
# 2️⃣ Scale the data
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -------------------------------
# 3️⃣ Prepare training data
# -------------------------------
look_back = 60  # Number of past days to use to predict next day

X = []
y = []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# LSTM expects 3D input [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -------------------------------
# 4️⃣ Build LSTM + ANN Model
# -------------------------------
model = Sequential()

# LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# ANN / Dense layers
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1))  # Predicting next closing price

model.compile(optimizer='adam', loss='mean_squared_error')

# -------------------------------
# 5️⃣ Train the model
# -------------------------------
model.fit(X, y, epochs=20, batch_size=32)

# -------------------------------
# 6️⃣ Predict on last 60 days
# -------------------------------
test_data = scaled_data[-look_back:]
X_test = []
X_test.append(test_data[:, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

print("Predicted next closing price:", predicted_price[0][0])
