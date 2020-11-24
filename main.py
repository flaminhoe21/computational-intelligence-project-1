import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as mp
import pickle
mp.style.use('fivethirtyeight')

s = datetime.datetime(2015,1,1)
e = datetime.datetime(2020,1,20)
net_data = web.DataReader('NFLX', data_source='yahoo', start=s, end=e)

net_data

net_data.shape

mp.figure(figsize=(12, 6))
mp.title('Netflix Stock-Price History')
mp.plot(net_data['Close'])
mp.xlabel('Date',fontsize=16)
mp.ylabel('Stock-Price in USD', fontsize=16)
mp.show()

netflix_table = net_data.filter(['Close'])
set_net_data = netflix_table.values
netflix_len_train_data = math.ceil(len(set_net_data) *.8)

net_stock_raster = MinMaxScaler(feature_range=(0, 1))
rasterized_data = net_stock_raster .fit_transform(set_net_data)

netflix_train_data = rasterized_data[0:netflix_len_train_data, :]
x_train = []
y_train = []
for i in range(60, len(netflix_train_data)):
    x_train.append(netflix_train_data[i-60:i, 0])
    y_train.append(netflix_train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

netflix_model = Sequential()
netflix_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
netflix_model.add(LSTM(units=50, return_sequences=False))
netflix_model.add(Dense(units=25))
netflix_model.add(Dense(units=1))

netflix_model.compile(optimizer='adam', loss='mean_squared_error')

netflix_model.fit(x_train, y_train, batch_size=1, epochs=1)

testing_netflix_data = rasterized_data[netflix_len_train_data - 60:, :]
x_test = []
y_test = set_net_data[netflix_len_train_data : , :]
for i in range(60,len(testing_netflix_data)):
    x_test.append(testing_netflix_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

netflix_predict = netflix_model.predict(x_test)
netflix_predict = net_stock_raster.inverse_transform(netflix_predict)

netflix_stock_rm = np.sqrt(np.mean(((netflix_predict - y_test)**2)))

train = netflix_table[:netflix_len_train_data]
valid = netflix_table[netflix_len_train_data:]
valid['Predictions'] = netflix_predict
mp.figure(figsize=(12, 6))
mp.title('Netflix Machine Learning Model')
mp.xlabel('Date', fontsize=16)
mp.ylabel('Netflix Stock-Price in USD', fontsize=16)
mp.plot(train['Close'])
mp.plot(valid[['Close', 'Predictions']])
mp.legend(['Training Model', 'Valid', 'Predictions'], loc='lower right')
mp.show()

valid


netflix_dataframe = web.DataReader('NFLX', data_source='yahoo', start=s, end=e)
new_df = netflix_dataframe.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = net_stock_raster.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict_nflx_price = netflix_model.predict(X_test)
predict_nflx_price = net_stock_raster.inverse_transform(predict_nflx_price)
print(predict_nflx_price)


s_test = datetime.datetime(2020,1,21)
e_test = datetime.datetime(2020,1,21)
netflix_dataframe2 = web.DataReader('NFLX', data_source='yahoo',
                              start=s_test, end=e_test)
print(netflix_dataframe2['Close'])

