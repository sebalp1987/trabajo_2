import STRING
import matplotlib.pyplot as plot
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

sns.set()

# Movilidad Urbana
df = pd.read_csv(STRING.movilidad_urbana, sep=';')

plot.barh(df['Mode'], df['Percentage'])
plot.gca().xaxis.set_major_formatter(PercentFormatter(1))
plot.ylabel('Transport Mode')
# plot.show()
plot.close()

# Weather condition
df = pd.read_csv(STRING.weather_condition, sep=';', encoding='latin1')
plot.plot(df['MONTH'], df['Average Temp (Cº)'], label='Tavg (Cº)')
plot.plot(df['Average Tmax (Cº)'], label='Tmax (Cº)')
plot.plot(df['Average Tmin (Cº)'], label='Tmin (Cº)')
plot.xticks(fontsize=10, rotation=45)
plot.ylabel('Temperature (Cº)')
plot.legend(loc="lower right")
# plot.show()
plot.close()

# Weather humidity
fig, ax1 = plot.subplots()
ax1.plot(df['MONTH'], df['HUMIDITY (%)'], label='Tavg (Cº)', color='lightcoral')
plot.gca().yaxis.set_major_formatter(PercentFormatter(1))
plot.xticks(fontsize=10, rotation=45)
plot.ylim(0)
ax1.set_ylabel('Humidity (%)', color='r')
ax2 = ax1.twinx()
ax2.plot(df['PREC (MM)'], label='Precipitation (mm)', color='mediumseagreen')
ax2.set_ylabel('Precipitation (mm)', color='g')
# fig.legend(loc="lower right")
ax2.grid(False)
plot.ylim(0)
# plot.show()
plot.close()

# Passengers
df = pd.read_csv(STRING.temporal_data, sep=';', encoding='latin1', date_parser=['DATE'])
_, tendencia_bus = sm.tsa.filters.hpfilter(df['PASSENGER_SUM_DAY'])
_, tendencia_train = sm.tsa.filters.hpfilter(df['PASSENGER_SUM_DAY_TRAIN'])
_, tendencia_metro = sm.tsa.filters.hpfilter(df['PASSENGER_SUM_DAY_METRO'])

df['tend_bus'] = tendencia_bus
df['tend_train'] = tendencia_train
df['tend_metro'] = tendencia_metro
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df = df.sort_values(by='DATE', ascending=True)
df['DATE'] = df['DATE'].apply(lambda x: x.date()).map(str)
plot.plot(df['DATE'], df['tend_bus'], label='BUS')
plot.plot(df['tend_train'], label='TRAIN')
plot.plot(df['tend_metro'], label='METRO')
plot.xticks(fontsize=10, rotation=45)
plot.ylabel('PASSENGERS')
plot.legend(loc="lower right")

xposition = ['2014-01-01', '2014-07-01', '2016-04-08']
for xv in xposition:
    plot.axvline(x=xv, color='k', linestyle='--')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
# plot.show()
plot.close()
df_2 = df[df['DATE'] > '2016-03-25']
plot.plot(df_2['DATE'], df_2['PASSENGER_SUM_DAY'])
plot.ylabel('PASSENGERS')
plot.axvline(x='2016-04-08', color='k', linestyle='--')
plot.xticks(fontsize=10, rotation=45)
# plot.show()


# Economy Variables
df = pd.read_csv(STRING.temporal_data, sep=';', encoding='latin1', date_parser=['DATE'])

fig, ax1 = plot.subplots()
xposition = ['2015-11-01']

ax1.plot(df['DATE'], df['DLS_LAST'], label='Change Rate (USD/ARS)', color='lightcoral', linestyle='-')
ax1.plot(df['UNEMPLOYMENT_RATE_GBA'], label='Unemployment Rate (%)', color='lightcoral')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df = df.sort_values(by='DATE', ascending=True)
df['DATE'] = df['DATE'].apply(lambda x: x.date()).map(str)
ax1.set_ylabel('Change Rate (USD/ARS) - Unemployment Rate (%)', color='r')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
for xv in xposition:
    plot.axvline(x=xv, color='k', linestyle='--')
ax2 = ax1.twinx()
ax2.plot(df['PRICE_LEVEL'], label='IPC', color='mediumseagreen')
ax2.plot(df['WAGE'], label='Wage Level', color='mediumseagreen', linestyle='--')
ax2.set_ylabel('Wage Index - CPI', color='g')

ax2.legend(loc="lower right")
ax1.legend(loc='lower left')
ax2.grid(False)
plot.show()
plot.close()

# PLOT FARES
df['NOMINAL_FARE'] = pd.Series(0, index=df.index)
df.loc[df['DATE'] < STRING.fares.date_fare_jan_14, 'NOMINAL_FARE'] = 1.81
df.loc[(df['DATE'] >= STRING.fares.date_fare_jan_14) & (
            df['DATE'] < STRING.fares.date_fare_jul_14), 'NOMINAL_FARE'] = 3.02
df.loc[(df['DATE'] >= STRING.fares.date_fare_jul_14) & (
            df['DATE'] < STRING.fares.date_fare_apr_16),  'NOMINAL_FARE'] = 3.49
df.loc[df['DATE'] >= STRING.fares.date_fare_apr_16, 'NOMINAL_FARE'] = 6.38

df['REAL_FARE'] = df['NOMINAL_FARE'] / df['DLS_LAST']


plot.plot(df['DATE'], df['NOMINAL_FARE'])
plot.plot(df['REAL_FARE'])
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.ylabel('PRICE')
plot.legend(loc="lower right")
xposition = ['2014-01-01', '2014-07-01', '2016-04-08']
for xv in xposition:
    plot.axvline(x=xv, color='k', linestyle='--')
plot.show()