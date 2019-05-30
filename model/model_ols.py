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

train = pd.read_csv(STRING.train_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test_detr, sep=';', encoding='latin1')

dict_values = STRING.variables_dict

# Variables Used
variable_used = config.params.get('linear_var')
variables = []
for col in variable_used:
    for col_d in train.columns.values.tolist():
        if col_d.endswith(col):
            variables.append(col_d)

variable_used = variables

# Replace by DICTIONARY
for col in train.columns.values.tolist():
    if col in dict_values:
        train = train.rename(columns={col: dict_values.get(col)})
        valid = valid.rename(columns={col: dict_values.get(col)})
        test = test.rename(columns={col: dict_values.get(col)})
        variable_used = [dict_values.get(col) if x==col else x for x in variable_used]


train['DATE'] = pd.to_datetime(train['DATE'])
train = train.set_index('DATE')
y = train[['Pax - Pax(-7)']]

vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(train[variable_used].drop('Trend', axis=1).values, i) for i in
              range(train[variable_used].drop('Trend', axis=1).shape[1])]
vif['features'] = train[variable_used].drop('Trend', axis=1).columns
print(vif)


# 1) INTERPRETABILIDAD
variable_used += ['AR7']
x_ols = train[variable_used].drop('AR7', axis=1)
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

''''
best_aic = []
ar_ma = []
i = 0

for ar in range(5, 10, 1):
    for ma in range(5, 10, 1):
        for sar in range(5, 10, 1):
            for mar in range(5, 10, 1):
                # VAR model to evaluate AIC best lags
                mod = sm.tsa.statespace.SARIMAX(endog=y, exog=x_ols, trend=None, order=(ar, 0, ma),
                                                seasonal_order=(sar, 0, mar, 7), jobs=-1)
                try:
                    results = mod.fit(disp=False)
                    std_error = results.bse
                    results = results.aic
                    print('iter ', i)
                    print(ar, ' ', ma, ' ', sar, ' ', mar)
                    print(results)
                    for se in range(0, len(std_error)-1, 1):
                        if np.isnan(std_error[se]):
                            results = 999999
                except:
                    results = 999999
                ar_ma.append([ar, ma, sar, mar])
                best_aic.append(results)
                i += 1
print(best_aic)
print(ar_ma)
ar_ma_coef = ar_ma[best_aic.index(min(best_aic))]
print('ARMA COEFFICIENTS', ar_ma_coef)
'''
# The PACF show a seasonal effect every 7 days (S.MA.7) No other autocorrelation effects.
drop_sarima = ['D1_DUMMY_GOVERN', 'D1 Exchange Rate (ARS/USD)',
               'D1 Economic Activity Index',
               'D1 Oil Final Price', 'D1 Temperature (ºC)',
               'Trend', 'Strong Wind',
               'D1_Vehicle Fleet', 'D1 Real Wage Index',
               'Workday']
x_sarima = train[variable_used].drop(drop_sarima, axis=1)
mod = sm.tsa.statespace.SARIMAX(endog=y, exog=x_sarima, trend=None, order=(0, 0, 0), seasonal_order=(0, 0, 2, 7))
results = mod.fit()
print(results.summary())
print(results.summary().as_latex())
# RESIDUALS
prediction = results.predict(exog=x_sarima)
print('R2 SARIMAX MODEL', r2_score(y.values, prediction))
res = pd.DataFrame(prediction, columns=['error'])
res = pd.concat([res, y], axis=1)
res['error'] = res['Pax - Pax(-7)'] - res['error']

# RESIDUAL ESTATIONARITY
sts.test_stationarity(res['error'], plot_show=False)
# RESIDUAL SERIAL CORRELATIONd
sts.serial_correlation(res['error'], plot_show=False)
fig, ax = plot.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res['error'], lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res['error'], lags=50, ax=ax[1])
plot.show()
plot.close()

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
train_x_sar = train[variable_used].drop(drop_sarima, axis=1)
valid_x_sar = valid[variable_used].drop(drop_sarima, axis=1)
test_x_sar = test[variable_used].drop(drop_sarima, axis=1)
train_x = train[variable_used].drop('AR7', axis=1)
valid_x = valid[variable_used].drop('AR7', axis=1)
test_x = test[variable_used].drop('AR7', axis=1)
train_y = train[['Pax - Pax(-7)']]
valid_y = valid[['Pax - Pax(-7)']]
test_y = test[['Pax - Pax(-7)']]

# OLS
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

# SARIMAX
modelA = sm.tsa.statespace.SARIMAX(endog=train_y, exog=train_x_sar, trend=None, order=(0, 0, 0), seasonal_order=(0, 0, 2, 7))
resA = modelA.fit()
prediction = resA.predict(start=0, end=len(valid_x_sar)-1, exog=valid_x_sar)

print('MSE SARIMAX ', mean_squared_error(valid_y.values, prediction))
print('MAE SARIMAX ', mean_absolute_error(valid_y.values, prediction))
print('R2 SARIMAX', r2_score(train_y.values, resA.predict(exog=train_x)))
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
valid_pred_sma = pd.DataFrame(resA.predict(start=0, end=len(valid_x_sar)-1, exog=valid_x_sar).values, columns=['SARIMAX_pred'],
                              index=valid.index)

valid_x['Nominal Fare'] = valid_x['Nominal Fare'] * 1.80
valid_x_sar['Nominal Fare'] = valid_x_sar['Nominal Fare'] * 1.80

valid_pred['OLS_pred80'] = pd.DataFrame(model_ols.predict(valid_x.values), index=valid.index)
valid_pred_sma['SARIMAX_pred80'] = pd.DataFrame(resA.predict(start=0, end=len(valid_x_sar)-1, exog=valid_x_sar).values,
                                                index=valid_x_sar.index)

test_pred = pd.DataFrame(model_ols.predict(test_x), columns=['OLS'], index=test_x.index)
test_pred_sma = pd.DataFrame(resA.forecast(len(test_x_sar), exog=test_x_sar).values,
                             columns=['SARIMAX'], index=test_x_sar.index)

# test_pred with older fare
test_x['Nominal Fare'] = pd.Series(1.94, index=test_x.index)
test_x_sar['Nominal Fare'] = pd.Series(1.94, index=test_x_sar.index)
test_pred_old = pd.DataFrame(model_ols.predict(test_x), columns=['OLS_SYNTETIC'], index=test_x.index)
test_pred_sma_old = pd.DataFrame(resA.forecast(len(test_x_sar), exog=test_x_sar).values,
                                 columns=['SARIMAX_SYNTETIC'],
                                 index=test_x_sar.index)
try:
    elasticity_file_valid = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1').set_index(
        valid_x.index)
    elasticity_file_test = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1').set_index(test_x.index)

except FileNotFoundError:
    elasticity_file_valid = pd.DataFrame(index=valid_x.index)
    elasticity_file_test = pd.DataFrame(index=test_x.index)
elasticity_file_valid = elasticity_file_valid.drop(
    [x for x in elasticity_file_valid.columns.values.tolist() if 'OLS' in x or 'SARIMAX' in x], axis=1)
elasticity_file_test = elasticity_file_test.drop(
    [x for x in elasticity_file_test.columns.values.tolist() if 'OLS' in x or 'SARIMAX' in x], axis=1)

elasticity_file_valid = pd.concat([elasticity_file_valid, valid_pred, valid_pred_sma], axis=1)
elasticity_file_test = pd.concat([elasticity_file_test, test_pred, test_pred_sma, test_pred_old, test_pred_sma_old],
                                 axis=1)
elasticity_file_valid.to_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1', index=False)
elasticity_file_test.to_csv(STRING.elasticity_file_test, sep=';', encoding='latin1', index=False)
