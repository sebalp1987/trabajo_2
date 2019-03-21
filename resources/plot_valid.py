import pandas as pd
import seaborn as sns
import STRING
import numpy as np
from scipy import stats

sns.set()

df = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1')
train = pd.read_csv(STRING.train_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
train_detr = pd.read_csv(STRING.train, sep=';', encoding='latin1')
valid_detr = pd.read_csv(STRING.valid, sep=';', encoding='latin1')

# Adicionamos el shift de vuelta
train_detr = train_detr[['DATE', 'PASSENGER_SUM_DAY']]
valid_detr = valid_detr[['DATE', 'PASSENGER_SUM_DAY']]

train = train[['DATE', 'PASSENGER_SUM_DAY']]
valid = valid[['DATE', 'PASSENGER_SUM_DAY']]
valid = pd.concat([train, valid], axis=0).reset_index(drop=True)
valid_detr = pd.concat([train_detr, valid_detr], axis=0).reset_index(drop=True)
valid_detr = valid_detr[8:].reset_index(drop=True)

valid['PASSENGER_SUM_DAY_ORIGINAL'] = np.exp(valid['PASSENGER_SUM_DAY'])*valid_detr['PASSENGER_SUM_DAY'].shift(7)
valid['PASSENGER_ORIGINAL_SHIFT'] = valid_detr['PASSENGER_SUM_DAY'].shift(7)
valid = valid[valid['DATE'] > '2015-10-14']

df = pd.concat([valid.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
df = df.dropna(axis=0)

df_file = pd.DataFrame(columns=['model', 'Elasticity', 'tstat', 'pvalue'])
for col in df.columns.values.tolist():
    if col != 'DATE' and not col.startswith('PASSENGER'):
        list_values = []
        print(col)
        elasticity = (df['PASSENGER_SUM_DAY_ORIGINAL'] - (np.exp(df[col]) * df['PASSENGER_ORIGINAL_SHIFT'])).mean()
        t, pvalue = stats.ttest_ind(df['PASSENGER_SUM_DAY_ORIGINAL'], np.exp(df[col]) * df['PASSENGER_ORIGINAL_SHIFT'])
        # If pvalue < 0.05 there is statical difference
        list_values.append([col, elasticity, t, pvalue])
        df_file = pd.concat([df_file, pd.DataFrame(list_values, columns=['model', 'Elasticity', 'tstat', 'pvalue'])],
                            axis=0)

df_file.to_csv(STRING.valid_comp_file, sep=';', encoding='latin1', index=False)
