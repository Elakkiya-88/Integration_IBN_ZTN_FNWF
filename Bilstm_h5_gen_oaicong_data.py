import pandas as pd

# Read the CSV file
# df = pd.read_csv("/kaggle/input/testbed-dataset/combined_throughput_2000.csv")
df =pd.read_csv("/kaggle/input/testbed-dataset/throughput_single_result_2000.csv") #/kaggle/input/testbed-dataset/throughput_result_with_2000bandwidths.csv")
# Extract 'Throughput (Kbps)' column and convert to integer list
throughput_list = df["Measured Throughput (Kbps)"].astype(int).tolist()

# Print the resulting list
# print(throughput_list)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from tensorflow.keras.regularizers import l2
import tensorflow as tf

flattened_sequence= throughput_list
# Step 1: Preprocessing the Data
states = flattened_sequence

# user_counts = [int(state[2:]) for state in states]
user_counts = states #[int(state[2:]) for state in states]

# Normalize the user_counts with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
user_counts_scaled = scaler.fit_transform(np.array(user_counts).reshape(-1, 1))

# Create sequences of a fixed length
sequence_length = 9
X = []
y = []

for i in range(len(user_counts_scaled) - sequence_length):
    X.append(user_counts_scaled[i:i + sequence_length])
    y.append(user_counts_scaled[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape X for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 2: Building the LSTM Model
bimodel = Sequential()

# First LSTM layer with Dropout regularization
bimodel.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(sequence_length, 1)))
bimodel.add(Dropout(0.3))

# Additional LSTM layers
bimodel.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.001))))
bimodel.add(Dropout(0.3))

# bimodel.add(Bidirectional(LSTM(200, return_sequences=True)))
# bimodel.add(Dropout(0.3))

bimodel.add(Bidirectional(LSTM(50)))

# Dense layers
# bimodel.add(Dense(512, activation='relu'))
# bimodel.add(Dense(256, activation='relu'))
bimodel.add(Dense(128, activation='relu'))
bimodel.add(Dense(64, activation='relu'))

# Output layer
bimodel.add(Dense(1))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.001)
bimodel.compile(optimizer=optimizer, loss='mean_squared_error')
#split the data into training, validation, and test test
X_train = X[0:1600]
y_train = y[0:1600]
X_val = X[1601:1800]
y_val = y[1601:1800]
X_test = X[1801:2100]
y_test = y[1801:2100]

# Step 3: Training the LSTM Model
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
# v = int(len(X_val))
# n = int(v / 2)
# history = bimodel.fit(X_train, y_train, epochs=40, validation_data=(X_val[1:n], y_val[1:n]), verbose=1)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)  # <-- Added ReduceLROnPlateau callback
history = bimodel.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val), callbacks=[lr_schedule], verbose=1)
# history = bimodel.fit(X_train, y_train, epochs=40, validation_data=(X_val[1:n], y_val[1:n]), callbacks=[lr_schedule], verbose=1)
# Plot training vs validation loss
plt.plot(history.history['loss'], label='Training Loss',linewidth=5)
plt.plot(history.history['val_loss'], label='Validation Loss',linewidth=5)
plt.xlabel("Epoch",fontweight='bold',fontsize=20)
plt.ylabel("Loss",fontweight='bold',fontsize=20)
# Increase font size of x and y tick values
plt.tick_params(axis='x', labelsize=18)  # Set font size for x-axis tick values
plt.tick_params(axis='y', labelsize=18)  # Set font size for y-axis tick values
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.legend()
plt.savefig("Loss Vs Epochs.eps",format='eps',dpi=1000)
plt.show()

# Save the LSTM model
bimodel.save('user_count_predictor_with_oai_dataset_with_UPF_congestion_.h5')

# Step 4: Predict using LSTM and Calculate Residuals
predicted_counts_scaled = bimodel.predict(X_test)
actual_counts_scaled = y_test
