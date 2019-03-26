import pandas as pd
import STRING
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plot
from scipy.stats import normaltest, shapiro
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import resources.temporal_statistics as sts
from config import config
sns.set()

df = pd.read_csv(STRING.temporal_data_detrend, sep=';', encoding='latin1')
train = pd.read_csv(STRING.train_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test_detr, sep=';', encoding='latin1')

dict_values = STRING.variables_dict

# Variables Used
variable_used = config.params.get('linear_var')
variables = []
for col in variable_used:
    for col_d in df.columns.values.tolist():
        if col_d.endswith(col):
            variables.append(col_d)

variable_used = variables

# Replace by DICTIONARY
for col in df.columns.values.tolist():
    if col in dict_values:
        df = df.rename(columns={col: dict_values.get(col)})
        train = train.rename(columns={col: dict_values.get(col)})
        valid = valid.rename(columns={col: dict_values.get(col)})
        test = test.rename(columns={col: dict_values.get(col)})
        variable_used = [dict_values.get(col) if x==col else x for x in variable_used]


df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

y = df[['Pax - Pax(-7)']]

print(variable_used)
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(df[variable_used].drop('Trend', axis=1).values, i) for i in
              range(df[variable_used].drop('Trend', axis=1).shape[1])]
vif['features'] = df[variable_used].drop('Trend', axis=1).columns
print(vif)


# 1) INTERPRETABILIDAD
x_ols = df[variable_used]
reg1 = sm.OLS(endog=y, exog=x_ols, missing='drop')
results = reg1.fit()
print(results.summary())
print(results.summary().as_latex())
prediction_ols = results.predict(x_ols)
print(r2_score(y.values, prediction_ols))

# RESIDUALS
res = pd.DataFrame(prediction_ols, columns=['error'])
res = pd.concat([res, y], axis=1)
res['error'] = res['Pax - Pax(-7)'] - res['error']
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
# plot.show()
plot.close()
# RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)
# RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=False)

# SARIMA MODEL
"""
Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly 
supports univariate time series data with a seasonal component.
"""
# We check PACF y ACF para MA y AR parameters.
sts.dw_test(x_ols, y, plot_show=False)

# ADD AR7
x_ols['AR7'] = y['Pax - Pax(-7)'].shift(7)
x_ols = x_ols.dropna(axis=0)
y = y[7:]
x_ols.dropna(axis=0)
# The PACF show a seasonal effect every 7 days (S.MA.7) No other autocorrelation effects.
mod = sm.tsa.statespace.SARIMAX(endog=y, exog=x_ols, trend=None, order=(0, 0, 0), seasonal_order=(0, 0, 1, 7))
results = mod.fit()
print(results.summary())
print(results.summary().as_latex())
# RESIDUALS
prediction = results.predict(start=0, end=len(y)-1, dynamic=True)
print('R2 SARIMAX MODEL', r2_score(y.values, prediction))
res = pd.DataFrame(prediction, columns=['error'])
res = pd.concat([res, y], axis=1)
res['error'] = res['Pax - Pax(-7)'] - res['error']
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
# plot.show()
plot.close()
# RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)
# RESIDUAL SERIAL CORRELATION
sts.serial_correlation(res['error'], plot_show=False)
stat, p = shapiro(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = normaltest(res['error'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# 2) PREDICCION
train_x = train[variable_used]
valid_x = valid[variable_used]
test_x = test[variable_used]
train_y = train[['Pax - Pax(-7)']]
valid_y = valid[['Pax - Pax(-7)']]
test_y = test[['Pax - Pax(-7)']]
model_ols = linear_model.LinearRegression()
model_ols.fit(train_x, train_y)
prediction = model_ols.predict(valid_x)
try:
    prediction_file = pd.read_csv(STRING.prediction_file, sep=';', encoding='latin1')
    if 'OLS' in prediction_file.columns.values.tolist():
        del prediction_file['OLS']
    prediction_file = pd.concat([prediction_file, pd.DataFrame(prediction, columns=['OLS'], index=prediction_file.index)], axis=1)
    prediction_file.to_csv(STRING.prediction_file, sep=';', encoding='latin1', index=False)
except FileNotFoundError:
    prediction_file = pd.DataFrame(prediction, columns=['OLS'])
    prediction_file = pd.concat([valid_y, prediction_file], axis=1)
    prediction_file.to_csv(STRING.prediction_file, sep=';', encoding='latin1', index=False)
    del prediction_file
print('MSE OLS ', mean_squared_error(valid_y.values, prediction))
print('MAE OLS ', mean_absolute_error(valid_y.values, prediction))
print('R2 OLS', r2_score(train_y.values, model_ols.predict(train_x)))

modelA = sm.tsa.statespace.SARIMAX(endog=train_y, exog=train_x, trend=None, order=(0, 0, 0), seasonal_order=(0, 0, 1, 7))
resA = modelA.fit()

prediction = resA.forecast(176, exog=valid_x)
print('MSE SARIMAX ', mean_squared_error(valid_y.values, prediction))
print('MAE SARIMAX ', mean_absolute_error(valid_y.values, prediction))
print('R2 SARIMAX', r2_score(train_y.values, resA.forecast(len(train_x), exog=train_x)))
try:
    prediction_file = pd.read_csv(STRING.prediction_file, sep=';', encoding='latin1')
    if 'SARIMAX' in prediction_file.columns.values.tolist():
        del prediction_file['SARIMAX']
    prediction_file = pd.concat([prediction_file, pd.DataFrame(prediction.values, columns=['SARIMAX'],
                                                               index=prediction_file.index)], axis=1)
    prediction_file.to_csv(STRING.prediction_file, sep=';', encoding='latin1', index=False)
except FileNotFoundError:
    prediction_file = pd.DataFrame(prediction, columns=['SARIMAX'])
    prediction_file = pd.concat([valid_y, prediction_file], axis=1)
    prediction_file.to_csv(STRING.prediction_file, sep=';', encoding='latin1', index=False)

# 3) ELASTICIDAD DEMANDA
valid_pred = pd.DataFrame(model_ols.predict(valid_x.values), columns=['OLS_pred'], index=valid.index)
valid_pred_sma = pd.DataFrame(resA.forecast(len(valid_x), exog=valid_x.values).values, columns=['SARIMAX_pred'],
                              index=valid.index)

valid_x['Nominal Fare'] = valid_x['Nominal Fare'] * 1.80
valid_pred['OLS_pred80'] = pd.DataFrame(model_ols.predict(valid_x.values), index=valid.index)
valid_pred_sma['SARIMAX_pred80'] = pd.DataFrame(resA.forecast(len(valid_x), exog=valid_x.values).values,
                                                index=valid.index)

test_pred = pd.DataFrame(model_ols.predict(test_x), columns=['OLS'], index=test_x.index)
test_pred_sma = pd.DataFrame(resA.forecast(len(test_x), exog=test_x.values).values, columns=['SARIMAX'], index=test_x.index)

# test_pred with older fare
test_x['Nominal Fare'] = pd.Series(1.94, index=test_x.index)
test_pred_old = pd.DataFrame(model_ols.predict(test_x), columns=['OLS_SYNTETIC'], index=test_x.index)
test_pred_sma_old = pd.DataFrame(resA.forecast(len(test_x), exog=test_x.values).values, columns=['SARIMAX_SYNTETIC'],
                                 index=test_x.index)
try:
    elasticity_file_valid = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1').set_index(
        valid_x.index)
    elasticity_file_test = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1').set_index(valid_x.index)

except FileNotFoundError:
    elasticity_file_valid = pd.DataFrame(index=valid_x.index)
    elasticity_file_test = pd.DataFrame(index=valid_x.index)
elasticity_file_valid = elasticity_file_valid.drop(
    [x for x in elasticity_file_valid.columns.values.tolist() if 'OLS' in x or 'SARIMAX' in x], axis=1)
elasticity_file_test = elasticity_file_test.drop(
    [x for x in elasticity_file_test.columns.values.tolist() if 'OLS' in x or 'SARIMAX' in x], axis=1)

elasticity_file_valid = pd.concat([elasticity_file_valid, valid_pred, valid_pred_sma], axis=1)
elasticity_file_test = pd.concat([elasticity_file_test, test_pred, test_pred_sma, test_pred_old, test_pred_sma_old],
                                 axis=1)
elasticity_file_valid.to_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1', index=False)
elasticity_file_test.to_csv(STRING.elasticity_file_test, sep=';', encoding='latin1', index=False)
