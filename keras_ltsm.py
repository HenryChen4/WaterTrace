# # Load the dataset (assuming it is in CSV format)
# df = pd.read_csv('lakepowell_monthly_25years.csv')

# # Extract the column containing the variable to be predicted (e.g., water level)
# data = df['average_level'].values

# # Scale the data to the range [0, 1]
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data.reshape(-1, 1))

# # Define the number of future steps to predict (25 years)
# n_steps = 300

# # Define the length of input sequences
# sequence_length = 40

# # Split the data into training and testing sets
# train_size = int(len(data) * 0.8)
# train_data = data[:train_size]
# test_data = data[train_size:]

# # Create sequences of data for training the LSTM
# def create_sequences(data, sequence_length):
#     x = []
#     y = []
#     for i in range(len(data) - sequence_length):
#         x.append(data[i:i + sequence_length])
#         y.append(data[i + sequence_length])
#     return np.array(x), np.array(y)

# x_train, y_train = create_sequences(train_data, sequence_length)
# x_test, y_test = create_sequences(test_data, sequence_length)

# # Define the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50))
# model.add(Dense(units=1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(x_train, y_train, epochs=100, batch_size=32, shuffle=False)

# # Generate predictions for the next 25 years
# input_seq = data[-sequence_length:]  # Take the last 'sequence_length' data points as input
# predictions = []
# for i in range(n_steps):
#     pred = model.predict(input_seq.reshape(1, sequence_length, 1))
#     predictions.append(pred[0][0])
#     input_seq = np.roll(input_seq, -1)  # Shift the input sequence by one position
#     input_seq[-1] = pred  # Append the new prediction to the input sequence

# # Inverse transform the predictions to the original scale
# predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# #------------------------------------------------------------------------------------------
