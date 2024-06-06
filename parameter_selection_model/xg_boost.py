from preprocessing import preprocessing_features, preprocessing_features_newer
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
import time

t1 = time.time()

X, y, groups = preprocessing_features_newer()
kf = GroupKFold(n_splits=10)

total_mae = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, y=y, groups=groups):
    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print(y_pred)
    print(y_val)

    mae = mean_absolute_error(y_val, y_pred)
    total_mae.append(mae)

t2 = time.time()


print(sum(total_mae) / len(total_mae))
print(np.var(total_mae))
print(t2 - t1)