import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import STRING
import numpy as np
from scipy import stats

sns.set()

df = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
test_detr = pd.read_csv(STRING.test, sep=';', encoding='latin1')
valid_detr = pd.read_csv(STRING.valid, sep=';', encoding='latin1')

# Adicionamos el shift de vuelta
test_detr = test_detr[['DATE', 'PASSENGER_SUM_DAY']]
valid_detr = valid_detr[['DATE', 'PASSENGER_SUM_DAY']]

test = test[['DATE', 'PASSENGER_SUM_DAY']]
valid = valid[['DATE', 'PASSENGER_SUM_DAY']]
test = pd.concat([valid, test], axis=0).reset_index(drop=True)
test_detr = pd.concat([valid_detr, test_detr], axis=0).reset_index(drop=True)
test_detr = test_detr[1:].reset_index(drop=True)
test['PASSENGER_SUM_DAY_ORIGINAL'] = np.exp(test['PASSENGER_SUM_DAY'])*test_detr['PASSENGER_SUM_DAY'].shift(7)
test['PASSENGER_ORIGINAL_SHIFT'] = test_detr['PASSENGER_SUM_DAY'].shift(7)
test = test[test['DATE'] > '2016-04-07']
print(test['PASSENGER_SUM_DAY_ORIGINAL'])
df = pd.concat([test.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
df = df.dropna(axis=0)


for col in df.columns:
    if col != 'DATE' and not col.startswith('PASSENGER') and not col.endswith('SYNTETIC') and col != 'LSTM' and col !='svr':
        print(np.exp(df[col])*df['PASSENGER_ORIGINAL_SHIFT'])
        plot.plot(np.exp(df[col])*df['PASSENGER_ORIGINAL_SHIFT'], label=col, linestyle='--')

plot.plot(df['DATE'], df['PASSENGER_SUM_DAY_ORIGINAL'], label='OBSERVED')
x = df['DATE'].tolist()
plot.xticks(x[::20], fontsize=10, rotation=45)
plot.legend(loc="lower right")
plot.ylabel('PASSENGERS (Pax)')
plot.show()

# ELASTICIDAD ESTIMADA ARCO
for col in df.columns:
    if col != 'DATE' and not col.startswith('PASSENGER') and not col.endswith('SYNTETIC'):
        print(col)

        # Ypred(t')
        df['Ypred(t\')_' + col] = (np.exp(df[col]) * df['PASSENGER_ORIGINAL_SHIFT'])

        # Ypred(t)
        df['Ypred(t)_' + col] = (np.exp(df[col + '_SYNTETIC']) * df['PASSENGER_ORIGINAL_SHIFT'])

        # Yobs(t') - Ypred(t) + |Yobs(t') - Ypred(t')| donde deberíamos tener que Yobs(t') - Ypred(t') --> 0 (Error)
        df['ElasticidadFinal_' + col] = df['PASSENGER_SUM_DAY_ORIGINAL'] - df['Ypred(t)_' + col] - (
                df['PASSENGER_SUM_DAY_ORIGINAL'] - df['Ypred(t\')_' + col])
        df['ElasticidadPuntual_' + col] = (df['ElasticidadFinal_' + col] / (3.54 - 1.94)) * 3.54 / df[
            'PASSENGER_SUM_DAY_ORIGINAL']
        _, pvalue_final = stats.ttest_ind(df['PASSENGER_SUM_DAY_ORIGINAL'], df['Ypred(t)_' + col] + (
                df['PASSENGER_SUM_DAY_ORIGINAL'] - df['Ypred(t\')_' + col]))
        df['pvalue_' + col] = pd.Series(pvalue_final, index=df.index)

        plot.plot(df['DATE'], df['PASSENGER_SUM_DAY_ORIGINAL'], label='OBSERVED(t\')', linestyle='--')
        plot.plot(df['Ypred(t\')_' + col], label='PREDICTED(t\')', linestyle='--')
        plot.plot(df['Ypred(t)_' + col], label='PREDICTED(t)',
                  linestyle='--')
        x = df['DATE'].tolist()
        plot.xticks(x[::20], fontsize=10, rotation=45)
        plot.legend(loc="lower right")
        plot.ylabel('Nominal Elasticity')
        plot.savefig(STRING.img_files + 'ELASTICITY_' + col + '.png')
        plot.show()
        plot.close()

        del df[col], df[col + '_SYNTETIC']


df.to_csv(STRING.elasticity_file, sep=';', encoding='latin1', index=False)


df['Expanded_mean_SARIMAX'] = df['ElasticidadPuntual_SARIMAX'].expanding().mean()
df['Expanded_mean_OLS'] = df['ElasticidadPuntual_OLS'].expanding().mean()
plot.plot(df['DATE'], np.abs(df['Expanded_mean_SARIMAX']), label='SARIMAX')
plot.plot(np.abs(df['Expanded_mean_OLS']), label='OLS')
x = df['DATE'].tolist()
plot.xticks(x[::5], fontsize=10, rotation=45)
plot.legend(loc="lower right")
plot.ylabel('Expanded Elasticity')
plot.show()