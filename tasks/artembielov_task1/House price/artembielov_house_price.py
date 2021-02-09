import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor


path = '...'
test = pd.read_csv(path + 'test.csv')
train = pd.read_csv(path + 'train.csv')

target = np.log(train.SalePrice)


all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))

all_data = all_data.drop(['Utilities'], axis=1)

useless = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
all_data.drop(useless, axis=1)


def encode(x):
    return 1 if x == 'Partial' else 0


all_data['enc_street'] = pd.get_dummies(all_data.Street, drop_first=True)
all_data['enc_condition'] = all_data.SaleCondition.apply(encode)

data_train = all_data[:train.shape[0]].select_dtypes(include=[np.number]).interpolate().dropna()
data_test = all_data[train.shape[0]:].select_dtypes(include=[np.number]).interpolate().dropna()

X_train = data_train
X_test = data_test
y = target

lr = LinearRegression(n_jobs=-1)
rd = Ridge(alpha=4.84)
rf = RandomForestRegressor(n_estimators=12, max_depth=3, n_jobs=-1)
gb = GradientBoostingRegressor(n_estimators=40, max_depth=2)
nn = MLPRegressor(hidden_layer_sizes=(90, 90), alpha=2.75)

model = StackingRegressor(regressors=[rf, gb, nn, rd], meta_regressor=lr)

model.fit(X_train, y)

y_pred = model.predict(X_train)

Y_pred = model.predict(X_test)

final_predictions = np.exp(Y_pred)
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = final_predictions
submission.to_csv('submission_v3.csv', index=False)
