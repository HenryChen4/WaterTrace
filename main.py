import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import csv

#chatGPT code
#------------------------------------------------------------------------------------------

# Load the dataset (assuming it is in CSV format)
df = pd.read_csv('lakepowell_monthly_25years.csv')

# Extract the column containing the variable to be predicted (e.g., water level)
data = df['average_level'].values

# Scale the data to the range [0, 1]
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# Define the number of future steps to predict (25 years)
n_steps = 300

# Define the length of input sequences (experiment with different lengths)
sequence_length = 60

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create sequences of data for training the LSTM
def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

# Data augmentation: Add noise to the training data
def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Apply data augmentation
augmented_train_data = add_noise(train_data, noise_level=0.01)

x_train, y_train = create_sequences(augmented_train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

# Define the LSTM model (experiment with different architectures)
model = Sequential()
model.add(Bidirectional(LSTM(units=150, return_sequences=True), input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=150, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=150)))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define a callback to reduce the learning rate when the validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# Train the model with validation split
history = model.fit(x_train, y_train, epochs=200, batch_size=32, shuffle=False, validation_split=0.1, callbacks=[reduce_lr])

# Generate predictions for the next 25 years
input_seq = data[-sequence_length:]  # Take the last 'sequence_length' data points as input
predictions = []
for i in range(n_steps):
    pred = model.predict(input_seq.reshape(1, sequence_length, 1))
    predictions.append(pred[0][0])
    input_seq = np.roll(input_seq, -1)  # Shift the input sequence by one position
    input_seq[-1] = pred  # Replace the last element of the input sequence with the new prediction

# Inverse transform
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

#------------------------------------------------------------------------------------------

# Graph the values
water_level_list = []
months_list = []

months = 0

with open('lakepowell_monthly_25years.csv', newline='') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        months += 1
        months_list.append(months)
        water_level_list.append(row[1])

fig, ax = plt.subplots()

water_level_list.pop(0)

water_level_list.extend(predictions)

for i in range(len(predictions)):
    months += 1
    months_list.append(months)

marker_setting = 'yo'
for i in range(months-1):
    if(i >= len(water_level_list) - len(predictions)):
        plt.plot(float(months_list[i]), float(water_level_list[i]), marker_setting, markersize = 1)
        marker_setting = 'go'
    else:
        plt.plot(float(months_list[i]), float(water_level_list[i]), marker_setting, markersize = 1)

ax.set_xlabel("months")
ax.set_ylabel("water level")

# Graph predicted data values

plt.show()