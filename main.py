import math
import json
from dateutil import parser
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt



data = json.load(open('dataset.json'))['prices']
years = {'2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'}
for inputs in data:
    for i in inputs:
        for j in years:
            if j in str(i):
                i = i[0:11]
                i = pd.to_datetime(parser.parse(i).strftime("%Y-%m-%d"))
                data[data.index(inputs)][0] = i
        if data[data.index(inputs)].index(i) == 2:
            data[data.index(inputs)][2] = int(data[data.index(inputs)][2])

df = pd.DataFrame(data, columns=['date', 'price', 'volume'])
df = df.set_index('date')

plt.figure(figsize=(16,8))
plt.title('Price History')
plt.plot(df['price'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price (USD)', fontsize=18)

df1 = df.filter(['price'])
dataset = df1.values
training_data_len = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []


for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)

test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

train = df1[:training_data_len]
valid = df1[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price (USD)', fontsize=18)
plt.plot(train["price"])
plt.plot(valid[['price', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
print(valid)

df2 = df1.copy()
last_60_days = df2[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
x_test = []
x_test.append(last_60_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)