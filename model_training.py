import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib

# Load preprocessed data (replace with your data path)
data_path = 'preprocessed_data.csv'
stock_data = pd.read_csv(data_path)

# Extract the target variable (Daily_Return)
y = stock_data['Daily_Return'].values

# Normalize the target variable
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define a function to create input sequences and labels for the LSTM model
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Create sequences and labels
sequence_length = 10
X, y = create_sequences(y_scaled, sequence_length)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = tf.keras.Sequential([tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)), tf.keras.layers.Dense(1)])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Save the trained model to a file
model_filename = 'stock_prediction_lstm_model'
model.save(model_filename, save_format='tf')