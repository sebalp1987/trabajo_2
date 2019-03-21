import pandas as pd
import STRING

clima = pd.read_csv(STRING.clima_processed, encoding='latin1', delimiter=';')
tren = pd.read_csv(STRING.tren_processed, encoding='latin1', delimiter=';')
subte = pd.read_csv(STRING.metro_processed, encoding='latin1', delimiter=';')
bus = pd.read_csv(STRING.bus_processed, encoding='latin1', delimiter=';')
dolar = pd.read_csv(STRING.dolar, delimiter=';', encoding='latin1')
wages = pd.read_csv(STRING.wages, delimiter=';', encoding='latin1')
inflation = pd.read_csv(STRING.inflation, delimiter=';', encoding='latin1')

dolar['DATE'] = pd.to_datetime(dolar['DATE'], format='%d/%m/%Y', errors='coerce')
wages['DATE'] = pd.to_datetime(wages['DATE'], format='%d/%m/%Y', errors='coerce')
bus['DATE'] = pd.to_datetime(bus['DATE'], format='%Y-%m-%d')
tren['DATE'] = pd.to_datetime(tren['DATE'], format='%Y-%m-%d')
subte['DATE'] = pd.to_datetime(subte['DATE'], format='%Y-%m-%d')
clima['DATE'] = pd.to_datetime(clima['DATE'], format='%Y-%m-%d')
bus = bus.fillna(0)
keys = ['DATE']

# DUMMY GOVERN
bus['DUMMY_GOVERN'] = pd.Series(0, index=bus.index)
print(bus['DATE'])
bus.loc[bus['DATE'] >= '2015-11-01', 'DUMMY_GOVERN'] = 1

bus['FARE_REAL_INDEX'] = pd.Series(0, index=bus.index)
bus['FARE_NOMINAL_INDEX'] = pd.Series(0, index=bus.index)

index_nominal = STRING.fares.indice_nominal
index_real = STRING.fares.indice_real

bus.loc[bus['DATE'] < STRING.fares.date_fare_jan_14, 'FARE_NOMINAL_INDEX'] = index_nominal[0]
bus.loc[(bus['DATE'] >= STRING.fares.date_fare_jan_14) & (
            bus['DATE'] < STRING.fares.date_fare_jul_14), 'FARE_NOMINAL_INDEX'] = index_nominal[1]
bus.loc[(bus['DATE'] >= STRING.fares.date_fare_jul_14) & (
            bus['DATE'] < STRING.fares.date_fare_apr_16),  'FARE_NOMINAL_INDEX'] = index_nominal[2]
bus.loc[bus['DATE'] >= STRING.fares.date_fare_apr_16, 'FARE_NOMINAL_INDEX'] = index_nominal[3]

bus.loc[bus['DATE'] < STRING.fares.date_fare_jan_14, 'FARE_REAL_INDEX'] = index_real[0]
bus.loc[(bus['DATE'] >= STRING.fares.date_fare_jan_14) & (
            bus['DATE'] < STRING.fares.date_fare_jul_14), 'FARE_REAL_INDEX'] = index_real[1]
bus.loc[(bus['DATE'] >= STRING.fares.date_fare_jul_14) & (
            bus['DATE'] < STRING.fares.date_fare_apr_16), 'FARE_REAL_INDEX'] = index_real[2]
bus.loc[bus['DATE'] >= STRING.fares.date_fare_apr_16, 'FARE_REAL_INDEX'] = index_real[3]

bus['GENERAL_STRIKE'] = pd.Series(0, index=bus.index)
bus.loc[bus['DATE'].isin(STRING.dates.paros), 'GENERAL_STRIKE'] = 1

bus = bus.drop_duplicates(subset=['DATE'], keep='first')

# MERGE DATAFRAME
# Train
'''
tren = tren[['DATE', 'PASSENGER_SUM_DAY', 'INCOME_SUM_DAY', 'PASSENGER_MAX_DAY', 'PASSENGER_MIN_DAY', 'PASSENGER_MEDIAN_DAY',
            'PASSENGER_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MIN_DAY', 'INCOME_MEDIAN_DAY', 'INCOME_AVG_DAY', 'FARE_AVG']]
'''
tren = tren[['DATE', 'PASSENGER_SUM_DAY', 'FARE_AVG']]

tren = tren.add_suffix('_TRAIN')
tren['DATE'] = tren['DATE_TRAIN']
del tren['DATE_TRAIN']

# Metro
'''
subte = subte[['DATE', 'PASSENGER_SUM_DAY', 'INCOME_SUM_DAY', 'PASSENGER_MAX_DAY', 'PASSENGER_MIN_DAY', 'PASSENGER_MEDIAN_DAY',
               'PASSENGER_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MIN_DAY', 'INCOME_MEDIAN_DAY', 'INCOME_AVG_DAY',
               'FARE_AVG']]
'''
subte = subte[['DATE', 'PASSENGER_SUM_DAY', 'FARE_AVG']]

subte = subte.add_suffix('_METRO')
subte['DATE'] = subte['DATE_METRO']
del subte['DATE_METRO']

date = bus[['DATE']]
date = date.drop_duplicates()
clima = pd.merge(date, clima, how='left', on='DATE')
clima = clima.sort_values(by=['DATE'], ascending=[True])
clima = clima.interpolate(limit_direction='backward')
bus = pd.merge(bus, tren, how='left', on='DATE')
bus = pd.merge(bus, subte, how='left', on='DATE')
bus = pd.merge(bus, clima.drop(['HUMIDITY_8_14_20(%)', 'WET_FOLIAGE(h)', 'STEAM_TENSION (hPa)',
                                'DEW (ºC)', 'COLD_UNITIES (h)', 'ATM_PRESSURE (hPa)',
                                'TEMP_GROUND_10cm', 'TEMP_MIN_OUTSIDE_1.5m', 'WIND_SPEED_10m(km/h)'], axis=1), how='left', on='DATE')

bus['DATE'] = pd.to_datetime(bus['DATE'], format='%Y-%m-%d')

# ECONOMIC INDICATORS: PBI, UNEMPLOYMENT, DOLAR VARIATION, POVERTY, SALARIO, INFLATION RATES, ESTMADOR MENSUAL
# INDUSTRIAL, ESTIMADOR MENSUAL ACTIVIDAD ECONOMICA

bus['YEAR'] = pd.DatetimeIndex(bus['DATE']).year
bus['MONTH'] = pd.DatetimeIndex(bus['DATE']).month

dolar = pd.merge(date, dolar[['DATE', 'DLS_LAST']], how='left', on='DATE')
dolar = dolar.sort_values(by=['DATE'], ascending=[True])
dolar = dolar.interpolate(limit_direction='backward')

print(dolar[dolar['DATE'] > '2016-01-26'])
bus = pd.merge(bus, dolar, how='left', on='DATE')
bus = pd.merge(bus, wages[['DATE', 'WAGE']], how='left', on='DATE')
bus = pd.merge(bus, inflation[['MONTH', 'YEAR', 'PRICE_LEVEL', 'UNEMPLOYMENT_RATE_GBA', 'NAFTA SUPER(CF)', 'EMAE', 'Vehicle Sells', 'Vehicle Transfers', 'Vehicle Fleet']],
               how='left', on=['MONTH', 'YEAR'])
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)


bus = bus.interpolate(limit_direction='backward')
print(bus.isnull().sum())
# bus = bus.dropna(axis=0, how='any')
del_var = ['PASSENGERS', 'INCOME']
for i in del_var:
    del bus[i]

bus['DLS_LAST'] = bus['DLS_LAST'].map(float)
bus['WAGE'] = bus['WAGE'].map(float)
bus['PRICE_LEVEL'] = bus['PRICE_LEVEL'].map(float)

drop_var = ['INCOME_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MEDIAN_DAY', 'INCOME_MIN_DAY',
            'INTERN_MAX_HOUR', 'INTERN_MEDIAN_HOUR', 'INTERN_MIN_HOUR', 'PASSENGER_AVG_DAY',
            'PASSENGER_MAX_DAY', 'PASSENGER_MEDIAN_DAY', 'PASSENGER_MIN_DAY']
bus = bus.drop(drop_var, axis=1)

for i in bus.columns.values.tolist():
    if 'FARE_AVG' in i or 'INCOME' in i:
        var_name_dls = i + '_DLS'
        var_name_wage = i + '_WAGE'
        var_name_inflation = i + '_IPC'
        bus[i] = bus[i].map(float)
        bus[var_name_dls] = bus[i] / bus['DLS_LAST']
        bus[var_name_wage] = bus[i] / bus['WAGE']
        bus[var_name_inflation] = bus[i] / bus['PRICE_LEVEL']

del dolar, wages, inflation

# FARE VARIATONS-RELATIVES-ETC
bus['FARE_BUS_TRAIN'] = bus['FARE_AVG'] / bus['FARE_AVG_TRAIN']
bus['FARE_BUS_METRO'] = bus['FARE_AVG'] / bus['FARE_AVG_METRO']
bus['FARE_METRO_TRAIN'] = bus['FARE_AVG_METRO'] / bus['FARE_AVG_TRAIN']

bus['PASSENGERS_BUS_TRAIN'] =bus['PASSENGER_SUM_DAY_TRAIN'] /  bus['PASSENGER_SUM_DAY']
bus['PASSENGERS_BUS_METRO'] =  bus['PASSENGER_SUM_DAY_METRO'] / bus['PASSENGER_SUM_DAY']
bus['PASSENGERS_METRO_TRAIN'] =  bus['PASSENGER_SUM_DAY_TRAIN'] / bus['PASSENGER_SUM_DAY']

bus = bus.reset_index(drop=True)

# WIND STRONG > 50 Viento Fuerte (Escala de Beaufort)
bus['WIND_STRONG'] = pd.Series(0, index=bus.index)
bus.loc[bus['WIND_SPEED_2m(km/h)'] > 50, 'WIND_STRONG'] = 1

bus['WIND_MAX_STRONG'] = pd.Series(0, index=bus.index)
bus.loc[bus['WIND_MAX_SPEED(km/h)'] > 50, 'WIND_MAX_STRONG'] = 1

date = date.sort_values(by=['DATE'], ascending=[True])
date = date.reset_index(drop=True)

date['TREND'] = pd.Series(date.index, index=date.index)
bus = pd.merge(bus, date, how='left', on='DATE')

bus['REAL_WAGE'] = bus['WAGE'] / bus['PRICE_LEVEL']
bus['REAL_FARE'] = bus['FARE_AVG'] / bus['PRICE_LEVEL']
bus['NAFTA_SUPER_CF'] = bus['NAFTA SUPER(CF)'].map(float) / bus['PRICE_LEVEL']

for i in bus.columns.values.tolist():
    if (i.startswith('DAY_') or i.startswith('YEAR_')or i.startswith('WEEKDAY_'))and(i != 'DAY_holiday'):
        del bus[i]

cols = bus.columns.values.tolist()
# bus = bus[['PASSENGER_SUM_DAY'] + [x for x in cols if not 'PASSENGER_SUM_DAY' in x]]
drop_var = ['PASSENGERS_METRO_TRAIN', 'PASSENGERS_BUS_METRO', 'PASSENGERS_BUS_TRAIN', 'PASSENGERS_BUS_TRAIN'
            , 'FARE_AVG_METRO_WAGE', 'FARE_AVG_METRO_DLS',
            'FARE_AVG_TRAIN_WAGE', 'FARE_AVG_TRAIN_DLS', 'INCOME_SUM_DAY_WAGE', 'INCOME_SUM_DAY_DLS', 'FARE_AVG_WAGE',
            'YEAR', 'FARE_AVG_METRO', 'FARE_AVG_TRAIN', 'FARE_AVG']
bus = bus.drop(drop_var, axis=1)
del bus['MONTH']
bus = bus.sort_values(by='DATE', ascending=True)
print(bus.columns.values.tolist())
bus.to_csv(STRING.temporal_data, sep=';', encoding='latin1', index=False)