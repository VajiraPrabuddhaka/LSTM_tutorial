import pandas
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import Utilities.myutil_regr as myutil

from pandas import Series

dataset = pandas.read_csv('dengue_labels_train.csv', usecols=[3], engine='python', skipfooter=3)

dataset1 = pandas.read_csv('dengue_features_train.csv')
dataset2 = pandas.read_csv('dengue_labels_train.csv')

iq, sj = myutil.split_dataset_by_city(dataset2)
iq = iq.drop(columns=['city', 'year', 'weekofyear'])
sj = sj.drop(columns=['city', 'year', 'weekofyear'])

dfall_iq, dfall_sj = myutil.split_dataset_by_city(dataset1)

dfall_iq = dfall_iq.drop(columns=['city', 'year', 'weekofyear', 'week_start_date'])
dfall_sj = dfall_sj.drop(columns=['city', 'year', 'weekofyear', 'week_start_date'])


dfall_iq['total_cases'] = Series(iq['total_cases'], index=dfall_iq.index)
dfall_sj['total_cases'] = Series(sj['total_cases'], index=dfall_iq.index)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dfall_iq.fillna(0, inplace=True)

# convert series to supervised learning
from pandas import DataFrame, concat
from sklearn.preprocessing import LabelEncoder

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = dfall_iq.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#print (reframed)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]], axis=1, inplace=True)
#reframed.drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], axis=1, inplace=True)
print(type(reframed))

# split into train and test sets
values = reframed.values
print(values.shape)
n_train_hours = int((values.shape)[0] * 0.67)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()


dataset_test = pandas.read_csv('dengue_features_test.csv')
iq_test, sj_test = myutil.split_dataset_by_city(dataset_test)
print ("fu")
iq_test = iq_test.drop(columns=['city', 'year', 'weekofyear', 'week_start_date'])

iq_test.fillna(0, inplace=True)

values_test_iq  = iq_test.values
values_test_iq = values_test_iq.astype('float32')

scaled_test = scaler.fit_transform(values_test_iq)


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]




# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
