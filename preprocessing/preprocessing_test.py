import STRING
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest, shapiro
import numpy as np

from resources import temporal_statistics as sts
from config import config

sns.set()

df = pd.read_csv(STRING.temporal_data, sep=';', encoding='latin1', date_parser=['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df = df[['PASSENGER_SUM_DAY'] + config.params.get('linear_var')]

# ESTACIONARIEDAD PASAJEROS

t_value, critical_value = sts.test_stationarity(df['PASSENGER_SUM_DAY'], plot_show=True)
dif = 0
while t_value > critical_value:
    dif += 1
    print(t_value)
    print(critical_value)
    t_value, critical_value = sts.test_stationarity(
        (df['PASSENGER_SUM_DAY'] - df['PASSENGER_SUM_DAY'].shift(dif)).dropna(axis=0), plot_show=False)
if dif > 0:
    df['D' + str(dif) + '_PASSENGER_SUM_DAY'] = df['PASSENGER_SUM_DAY'] - df['PASSENGER_SUM_DAY'].shift(dif)
    del df['PASSENGER_SUM_DAY']

# DECOMPOSE SERIE ADDITIVE (DETREND / DESEASONALIZE)
"""
Depending on the nature of the trend and seasonality, a time series can be modeled as an additive or multiplicative, 
wherein, each observation in the series can be expressed as either a sum or a product of the components:

Additive time series:
Value = Base Level + Trend + Seasonality + Error

Multiplicative Time Series:
Value = Base Level x Trend x Seasonality x Error

If you look at the residuals of the multiplicatve decomposition closely, it has some pattern left over. The additive
decomposition, however, looks quite random which is good. So ideally, additive decomposition should be preferred 
for this particular series.
"""

result_add = seasonal_decompose(df['PASSENGER_SUM_DAY'], model='additive', extrapolate_trend='freq')
result_mul = seasonal_decompose(df['PASSENGER_SUM_DAY'], model='multiplicative', extrapolate_trend='freq')
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
plot.show()
plot.close()
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plot.show()
plot.close()


# DESEASON THE VARIABLE
# Se ve claramente una tendencia cada 7 días de seasonality
df['PASSENGER_SUM_DAY'] = np.log(df['PASSENGER_SUM_DAY']) - np.log(df['PASSENGER_SUM_DAY'].shift(7))
df = df.dropna(axis=0)
sts.test_stationarity(df['PASSENGER_SUM_DAY'], plot_show=True)


# ESTACIONARIEDAD DEMAS VARIABLES
for col in df.columns.values.tolist():
    print(col)
    print(df[col])
    df[col] = df[col].map(float)
bool_cols = [col for col in df
             if df[[col]].dropna().isin([0, 1]).all().values]
for cols in df.drop(['PASSENGER_SUM_DAY', 'TREND', 'PRECIPITACION-MM', 'FARE_NOMINAL_INDEX'
                     ] + bool_cols, axis=1).columns.values.tolist():
    print(cols)
    # df[cols] = np.log(df[cols]) log-linear
    df[cols] = df[cols].map(float)
    dif = 0

    try:
        t_value, critical_value = sts.test_stationarity(df[cols], plot_show=False)

        while t_value > critical_value:
            dif += 1
            print(t_value)
            print(critical_value)
            print(dif)
            t_value, critical_value = sts.test_stationarity((df[cols] - df[cols].shift(dif)).dropna(axis=0),
                                                            plot_show=False)
    except:
        pass
    if dif > 0:
        df['D' + str(dif) + '_' + cols] = df[cols] - df[cols].shift(dif)
        del df[cols]

# Detrend-Deseasonalize: If additive Y - season-trend, If multiplicative Y/ /
'''
df['PASSENGER_SUM_DAY'] = df['PASSENGER_SUM_DAY'] - result_add.seasonal - result_add.trend
plot.plot(df['PASSENGER_SUM_DAY'])
plot.title('PASSENGERS DETREND-DESEASONALIZED', fontsize=16)
plot.show()
plot.close()
'''
# AUTOCORRELACION PASAJEROS (DW)
df = df.dropna(axis=0)
sts.dw_test(df.drop('PASSENGER_SUM_DAY', axis=1), df[['PASSENGER_SUM_DAY']])

# AR PARAMETERS
lags = 7
'''
for i in range(lags, lags + 1, 1):
    df['AR_' + str(i)] = df['PASSENGER_SUM_DAY'].shift(i)
'''

# MA PARAMETERS
# Calculate the moving average. That is, take
# the first 7 values, average them,
# then drop the first and add the third, etc.
ma = 7
'''
for i in range(1, ma + 1, 1):
    df['MA_' + str(i)] = df['PASSENGER_SUM_DAY'].rolling(window=i).mean()
sts.dw_test(df.drop('PASSENGER_SUM_DAY', axis=1), df[['PASSENGER_SUM_DAY']], plot_show=False)
'''



# ESTACIONARIEDAD RESIDUOS
df = df.dropna(axis=0)
error = sts.get_residuals(df.drop('PASSENGER_SUM_DAY', axis=1), df['PASSENGER_SUM_DAY'])
sts.test_stationarity(error['error'], plot_show=True)

# AUTOCORRELACION RESIUDOS
"""
We would not expect there to be any correlation between the residuals. This would be shown by autocorrelation scores 
being below the threshold of significance (dashed and dotted horizontal lines on the plot).

A significant autocorrelation in the residual plot suggests that the model could be doing a better job of incorporating 
the relationship between observations and lagged observations, called autoregression.
"""
sts.serial_correlation(error)
'''
plot.close()
plot.hist(error)
plot.show()
'''

# NORMALIDAD RESIDUOS
stat, p = shapiro(error)
print('Statistics=%.3f, p=%.3f' % (stat, p))
stat, p = normaltest(error)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')




df.reset_index(drop=False).to_csv(STRING.temporal_data_detrend, index=False, sep=';', encoding='latin1')

