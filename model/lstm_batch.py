import pandas as pd
import STRING
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from config import config

sns.set()
np.random.seed(42)
batch_size = 50

df = pd.read_csv(STRING.temporal_data_detrend, sep=';', encoding='latin1')
train = pd.read_csv(STRING.train_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test_detr, sep=';', encoding='latin1')
df_file = pd.DataFrame(columns=['model', 'params', 'coef', 'mse', 'mae', 'r2'])

df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

y = df[['PASSENGER_SUM_DAY']]
variable_used = config.params.get('linear_var')
variables = []
for col in variable_used:
    for col_d in df.columns.values.tolist():
        if col_d.endswith(col):
            variables.append(col_d)

variable_used = variables
x = df[variable_used]

train_x = train[variable_used]
valid_x = valid[variable_used]
test_x = test[variable_used]
train_y = train[['PASSENGER_SUM_DAY']]
valid_y = valid[['PASSENGER_SUM_DAY']]
test_y = test[['PASSENGER_SUM_DAY']]
train_x['TEST'] = pd.Series(0, index=train_x.index)
valid_x['TEST'] = pd.Series(1, index=valid_x.index)
test_x['TEST'] = pd.Series(2, index=test_x.index)

y = pd.concat([train_y, valid_y, test_y], axis=0)
x = pd.concat([train_x, valid_x, test_x], axis=0)


# We add the S.MA7
def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)


ma7 = calcSma(y['PASSENGER_SUM_DAY'], smaPeriod=8).tolist()
x['AR7'] = y['PASSENGER_SUM_DAY'].shift(7)
x['s.AR7'] = y['PASSENGER_SUM_DAY'] - y['PASSENGER_SUM_DAY'].shift(7)
x['MA7'] = pd.DataFrame(ma7, index=x.index)
x['s.MA7'] = y['PASSENGER_SUM_DAY'] - x['MA7']

x = x.dropna(axis=0)
y = y[7:]
x = pd.concat([x, y], axis=1)
# NORMALIZE
df_c = x[['TEST']].reset_index(drop=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
cols = x.drop(['TEST'], axis=1).columns.values.tolist()
print(cols)
df = scaler.fit_transform(x.drop(['TEST'], axis=1))
df = pd.DataFrame(df, columns=cols)
df = pd.concat([df_c, df], axis=1)
train_y = df[df['TEST'] == 0]
valid_y = df[df['TEST'] == 1]
test_y = df[df['TEST'] == 2]
train_y = train_y[['PASSENGER_SUM_DAY']]
valid_y = valid_y[['PASSENGER_SUM_DAY']]
test_y = test_y[['PASSENGER_SUM_DAY']]
train_x = df[df['TEST'] == 0].drop(['TEST', 'PASSENGER_SUM_DAY'], axis=1)
valid_x = df[df['TEST'] == 1].drop(['TEST', 'PASSENGER_SUM_DAY'], axis=1)
test_x = df[df['TEST'] == 2].drop(['TEST', 'PASSENGER_SUM_DAY'], axis=1)

train_batch = int(math.floor(train_x.shape[0] / batch_size) * batch_size)
valid_batch = int(math.floor(valid_x.shape[0] / batch_size) * batch_size)
test_batch = int(math.floor(test_x.shape[0] / batch_size) * batch_size)

train_x, train_y = train_x[:train_batch], train_y[:train_batch]
valid_x, valid_y = valid_x[:valid_batch], valid_y[:valid_batch]
test_x, test_y = test_x[:test_batch], test_y[:test_batch]

train_x = np.reshape(train_x.values, (train_x.values.shape[0], 1, train_x.values.shape[1]))
valid_x_re = np.reshape(valid_x.values, (valid_x.values.shape[0], 1, valid_x.values.shape[1]))
test_x_re = np.reshape(test_x.values, (test_x.values.shape[0], 1, test_x.values.shape[1]))

# MODEL
model = Sequential()
model.add(LSTM(30, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), activation='tanh',
               stateful=True,
               return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(30, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(30, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_x.shape))
print("Actual output:" + str(train_y.shape))
early_stopping = EarlyStopping(patience=2)
for i in range(100):
    model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, callbacks=[early_stopping],
              shuffle=False,
              validation_data=(valid_x_re, valid_y))
    model.reset_states()

# PREDICT
# Valid
valid_predict = model.predict(np.reshape(valid_x.values, (valid_x.values.shape[0], 1, valid_x.values.shape[1])),
                              batch_size=batch_size)
model.reset_states()
valid_x_ = valid_x_re.reshape(valid_x_re.shape[0], valid_x_re.shape[2])
valid_predict = np.concatenate((valid_predict, valid_x_), axis=1)
inv_predict = scaler.inverse_transform(valid_predict)
valid_predict = inv_predict[:, 0]

valid_y_ = valid_y.values.reshape(len(valid_y), 1)
inv_y = np.concatenate((valid_y_, valid_x_), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid_true = inv_y[:, 0]

# Valid25
valid_x['FARE_REAL_INDEX'] = valid_x['FARE_REAL_INDEX'] * 1.025
valid_predict25 = model.predict(np.reshape(valid_x.values, (valid_x.values.shape[0], 1, valid_x.values.shape[1])),
                                batch_size=batch_size)
model.reset_states()
valid_x_ = valid_x_re.reshape(valid_x_re.shape[0], valid_x_re.shape[2])
valid_predict25 = np.concatenate((valid_predict25, valid_x_), axis=1)
inv_predict = scaler.inverse_transform(valid_predict25)
valid_predict25 = inv_predict[:, 0]

valid_y_ = valid_y.values.reshape(len(valid_y), 1)
inv_y = np.concatenate((valid_y_, valid_x_), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid_true25 = inv_y[:, 0]

# Valid80
valid_x['FARE_NOMINAL_INDEX'] = valid_x['FARE_NOMINAL_INDEX'] * 1.80
valid_predict80 = model.predict(np.reshape(valid_x.values, (valid_x.values.shape[0], 1, valid_x.values.shape[1])),
                                batch_size=batch_size)
model.reset_states()
valid_x_ = valid_x_re.reshape(valid_x_re.shape[0], valid_x_re.shape[2])
valid_predict80 = np.concatenate((valid_predict80, valid_x_), axis=1)
inv_predict = scaler.inverse_transform(valid_predict80)
valid_predict80 = inv_predict[:, 0]

valid_y_ = valid_y.values.reshape(len(valid_y), 1)
inv_y = np.concatenate((valid_y_, valid_x_), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid_true80 = inv_y[:, 0]

# Test
test_predict = model.predict(np.reshape(test_x.values, (test_x.values.shape[0], 1, test_x.values.shape[1])),
                             batch_size=batch_size)
model.reset_states()
test_x = test_x_re.reshape(test_x_re.shape[0], test_x_re.shape[2])
test_predict = np.concatenate((test_predict, test_x), axis=1)
inv_predict = scaler.inverse_transform(test_predict)
test_predict = inv_predict[:, 0]

test_y = test_y.values.reshape(len(test_y), 1)
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
test_true = inv_y[:, 0]

# PREDICTIONS
valid_predict = pd.DataFrame(valid_predict, columns=['predict'])
valid_predict25 = pd.DataFrame(valid_predict25, columns=['predict'])
valid_predict80 = pd.DataFrame(valid_predict80, columns=['predict'])
test_predict = pd.DataFrame(test_predict, columns=['LSTM'])

valid_true = pd.DataFrame(valid_true, columns=['true_values'])
test_true = pd.DataFrame(test_true, columns=['true_values'])

# MSE
mse = mean_squared_error(valid_y, valid_predict)
mae = mean_absolute_error(valid_y, valid_predict)
r2 = r2_score(valid_y, valid_predict)
params = [['LSTM', '', '', mse, mae, r2]]
df_file = pd.concat([df_file, pd.DataFrame(params, columns=['model', 'params', 'coef', 'mse', 'mae', 'r2'])],
                    axis=0)

# Elasticidad
prediction = pd.DataFrame()
print(valid_predict)
prediction['LSTM'] = valid_predict['predict']
prediction['LSTM_pred25'] = valid_predict25
prediction['LSTM_pred80'] = valid_predict80

elasticity_file_valid = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1')
elasticity_file_test = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1')

elasticity_file_valid = elasticity_file_valid.drop(
    [x for x in elasticity_file_valid.columns.values.tolist() if 'LSTM' in x], axis=1)
elasticity_file_test = elasticity_file_test.drop(
    [x for x in elasticity_file_test.columns.values.tolist() if 'LSTM' in x], axis=1)

elasticity_file_valid = pd.concat([elasticity_file_valid, prediction], axis=1)
elasticity_file_test = pd.concat([elasticity_file_test, test_predict], axis=1)
elasticity_file_valid.to_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1', index=False)
elasticity_file_test.to_csv(STRING.elasticity_file_test, sep=';', encoding='latin1', index=False)

print(df_file)
