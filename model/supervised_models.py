import pandas as pd
import STRING
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plot
from scipy.stats import normaltest, shapiro
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
import graphviz

import resources.temporal_statistics as sts
from config import config
sns.set()

df = pd.read_csv(STRING.temporal_data_detrend, sep=';', encoding='latin1')
train = pd.read_csv(STRING.train_detr, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid_detr, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test_detr, sep=';', encoding='latin1')
df_file = pd.DataFrame(columns=['model', 'params', 'coef', 'mse', 'mae', 'r2'])

df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

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
        variable_used = [dict_values.get(col) if x == col else x for x in variable_used]

y = df[['Pax - Pax(-7)']]
train_x = train[variable_used]
valid_x = valid[variable_used]
test_x = test[variable_used]
train_y = train[['Pax - Pax(-7)']]
valid_y = valid[['Pax - Pax(-7)']]
test_y = test[['Pax - Pax(-7)']]
train_x['TEST'] = pd.Series(0, index=train_x.index)
valid_x['TEST'] = pd.Series(1, index=valid_x.index)
test_x['TEST'] = pd.Series(2, index=test_x.index)
train_y['TEST'] = pd.Series(0, index=train_y.index)
valid_y['TEST'] = pd.Series(1, index=valid_y.index)
test_y['TEST'] = pd.Series(2, index=test_y.index)

y = pd.concat([train_y, valid_y, test_y], axis=0)
x = pd.concat([train_x, valid_x, test_x], axis=0)


# We add the S.MA7
def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)


ma7 = calcSma(y['Pax - Pax(-7)'], smaPeriod=8).tolist()
x['AR7'] = y['Pax - Pax(-7)'].shift(7)
x['s.AR7'] = y['Pax - Pax(-7)'] - y['Pax - Pax(-7)'].shift(7)
x['MA7'] = pd.DataFrame(ma7, index=x.index)
x['s.MA7'] = y['Pax - Pax(-7)'] - x['MA7']
del x['AR7']
del x['MA7']

x = x.dropna(axis=0)
y = y[7:]
train_x = x[x['TEST'] == 0].drop('TEST', axis=1)
valid_x = x[x['TEST'] == 1].drop('TEST', axis=1)
test_x = x[x['TEST'] == 2].drop('TEST', axis=1)
train_y = y[y['TEST'] == 0].drop('TEST', axis=1)
valid_y = y[y['TEST'] == 1].drop('TEST', axis=1)
test_y = y[y['TEST'] == 2].drop('TEST', axis=1)

# ELASTIC NET
# 1) Interpretabilidad
file_model = linear_model.ElasticNetCV(cv=5).fit(train_x, train_y)
plot.figure()
plot.plot(file_model.alphas_, file_model.mse_path_, ':')  # este plotea los alphas y los mse
plot.plot(file_model.alphas_, file_model.mse_path_.mean(axis=-1), label='Average MSE Across Folds',
          linewidth=2)  # Este le da el formato y el label al ploteo de MSE
plot.axvline(file_model.alpha_, linestyle='--', label='CV Estimate of Best Alpha')
plot.semilogx()
plot.legend()
ax = plot.gca()
ax.invert_xaxis()
plot.xlabel('alpha')
plot.ylabel('Mean Square Errror')
plot.axis('tight')
plot.show()

alphaStar = file_model.alpha_
print('alpha that Min CV Error ', alphaStar)
print('Minimum MSE ', min(file_model.mse_path_.mean(axis=-1)))

enet = linear_model.ElasticNet(alpha=alphaStar, l1_ratio=1, fit_intercept=False, normalize=False)
enet.fit(train_x, train_y)
prediction = enet.predict(valid_x)
print(valid_x.columns)
print(file_model.coef_)

# 2) Prediccion MSE
mse = mean_squared_error(valid_y.values, prediction)
mae = mean_absolute_error(valid_y.values, prediction)
r2 = r2_score(train_y.values, file_model.predict(train_x))
params = [['enet', enet, file_model.coef_, mse, mae, r2]]
df_file = pd.concat([df_file, pd.DataFrame(params, columns=['model', 'params', 'coef', 'mse', 'mae', 'r2'])], axis=0)

# 3) Elasticidad
prediction = pd.DataFrame(prediction, columns=['ENET_pred'], index=valid_x.index)
valid_x['Nominal Fare'] = valid_x['Nominal Fare'] * 1.80
prediction['ENET_pred80'] = pd.DataFrame(enet.predict(valid_x), index=valid_x.index)
test_pred = pd.DataFrame(enet.predict(test_x), columns=['ENET'], index=test_x.index)

# test_pred with older fare
test_x['Nominal Fare'] = pd.Series(1.94, index=test_x.index)
test_pred['ENET_SYNTETIC'] = pd.DataFrame(enet.predict(test_x), index=test_x.index)

try:
    elasticity_file_valid = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1').set_index(valid_x.index)
    elasticity_file_test = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1').set_index(valid_x.index)

except FileNotFoundError:
    elasticity_file_valid = pd.DataFrame(index=valid_x.index)
    elasticity_file_test = pd.DataFrame(index=valid_x.index)
print(elasticity_file_valid)
elasticity_file_valid = elasticity_file_valid.drop(
        [x for x in elasticity_file_valid.columns.values.tolist() if 'ENET' in x], axis=1)
elasticity_file_test = elasticity_file_test.drop(
        [x for x in elasticity_file_test.columns.values.tolist()  if 'ENET' in x], axis=1)

elasticity_file_valid = pd.concat([elasticity_file_valid, prediction], axis=1)
elasticity_file_test = pd.concat([elasticity_file_test, test_pred], axis=1)
elasticity_file_valid.to_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1', index=False)
elasticity_file_test.to_csv(STRING.elasticity_file_test, sep=';', encoding='latin1', index=False)


# OTHER SUPERVISED MODELS
valid_x['Nominal Fare'] = valid_x['Nominal Fare'] / 1.80
test_x['Nominal Fare'] = pd.Series(3.54, index=test_x.index)

min_sample_leaf = round((len(train_x.index)) * 0.01)
min_sample_split = min_sample_leaf * 10

# Decision Tree
file_model_tree = tree.DecisionTreeRegressor(criterion=config.params.get('criterion'),
                                             max_depth=config.params.get('max_depth'),
                                             min_samples_split=min_sample_split, min_samples_leaf=min_sample_leaf,
                                             max_features=config.params.get('max_features'),
                                             random_state=config.params.get('random_state'))
file_model_tree.fit(train_x, train_y)
dot_data = tree.export_graphviz(file_model_tree, out_file=None, feature_names=train_x.columns,
                                class_names='Pax - Pax(-7)',
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render(STRING.img_files + 'tree_regressor')

# SVR

param_grid = [
  {'C': [1, 10, 100, 1000], 'tol':[0.001, 0.01], 'random_state':[config.params.get('random_state')],
                                                                 'epsilon':[0, 0.1, 0.2, 0.5]}
 ]

model_svr = GridSearchCV(LinearSVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
model_svr.fit(train_x, train_y)
print(model_svr.cv_results_['mean_test_score'], model_svr.cv_results_['params'])
print(model_svr.best_params_)

# Others
models = {'tree': file_model_tree,
          'bag': BaggingRegressor(base_estimator=file_model_tree, n_estimators=config.params.get('n_estimators'),
                                  max_samples=0.25,
                                  max_features=config.params.get('max_features'),
                                  bootstrap=config.params.get('bootstrap'),
                                  random_state=config.params.get('random_state')),
          'rf': RandomForestRegressor(criterion=config.params.get('criterion'),
                                      n_estimators=config.params.get('n_estimators'),
                                      max_depth=config.params.get('max_depth'), min_samples_split=min_sample_split,
                                      min_samples_leaf=min_sample_leaf, max_features='sqrt',
                                      bootstrap=config.params.get('bootstrap'),
                                      random_state=config.params.get('random_state')
                                      ),
          'gb': GradientBoostingRegressor(learning_rate=0.001, criterion='friedman_mse',
                                          min_samples_split=min_sample_split,
                                          n_estimators=config.params.get('n_estimators'),
                                          min_samples_leaf=min_sample_leaf, max_depth=config.params.get('max_depth'),
                                          random_state=config.params.get('random_state'),
                                          max_features=config.params.get('max_features')),
          'svr': LinearSVR(epsilon=0.0, tol=0.001, C=1.0, random_state=42)
          }


for name, file_model in models.items():
    print(name)
    # Interpretabilidad
    file_model.fit(train_x, train_y)
    prediction = file_model.predict(valid_x)
    if name == 'bag':
        feature_importance = np.mean([
            tree.feature_importances_ for tree in file_model.estimators_
        ], axis=0)
        feature_importance = feature_importance / feature_importance.max()
    elif name == 'svr':
        feature_importance = file_model.coef_.ravel()

    else:
        feature_importance = file_model.feature_importances_
        feature_importance = feature_importance / feature_importance.max()
    print(train_x.columns)
    print(valid_x.columns)
    print(feature_importance)
    # MSE
    mse = mean_squared_error(valid_y.values, prediction)
    mae = mean_absolute_error(valid_y.values, prediction)
    r2 = r2_score(train_y.values, file_model.predict(train_x))
    params = [[name, file_model, feature_importance, mse, mae, r2]]
    df_file = pd.concat([df_file, pd.DataFrame(params, columns=['model', 'params', 'coef', 'mse', 'mae', 'r2'])],
                        axis=0)

    # Elasticidad
    prediction = pd.DataFrame(prediction, columns=[name + '_pred'], index=valid_x.index)
    valid_x['Nominal Fare'] = valid_x['Nominal Fare'] * 1.80
    prediction[name + '_pred80'] = pd.DataFrame(file_model.predict(valid_x), index=valid_x.index)
    test_pred = pd.DataFrame(file_model.predict(test_x), columns=[name], index=test_x.index)

    # test_pred with older fare
    test_x['Nominal Fare'] = pd.Series(1.94, index=test_x.index)
    test_pred[name + '_SYNTETIC'] = pd.DataFrame(file_model.predict(test_x), index=test_x.index)

    try:
        elasticity_file_valid = pd.read_csv(STRING.elasticity_file_valid, sep=';', encoding='latin1').set_index(
            valid_x.index)
        elasticity_file_test = pd.read_csv(STRING.elasticity_file_test, sep=';', encoding='latin1').set_index(
            valid_x.index)

    except FileNotFoundError:
        elasticity_file_valid = pd.DataFrame(index=valid_x.index)
        elasticity_file_test = pd.DataFrame(index=valid_x.index)
        
    elasticity_file_valid = elasticity_file_valid.drop(
        [x for x in elasticity_file_valid.columns.values.tolist() if name in x], axis=1)
    elasticity_file_test = elasticity_file_test.drop(
        [x for x in elasticity_file_test.columns.values.tolist() if name in x], axis=1)

    elasticity_file_valid = pd.concat([elasticity_file_valid, prediction], axis=1)
    elasticity_file_test = pd.concat([elasticity_file_test, test_pred], axis=1)
    elasticity_file_valid.to_csv(STRING.elasticity_file_valid, sep=';',index=False)
    elasticity_file_test.to_csv(STRING.elasticity_file_test, sep=';', index=False)

    valid_x['Nominal Fare'] = valid_x['Nominal Fare'] / 1.80
    test_x['Nominal Fare'] = pd.Series(3.54, index=test_x.index)


df_file.to_csv(STRING.param_file, sep=';', encoding='latin1', index=False)
