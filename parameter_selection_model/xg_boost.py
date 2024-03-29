from preprocessing import preprocessing_features
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time

t1 = time.time()

X, y = preprocessing_features()
kf = KFold(n_splits=10, shuffle=True)

total_mse = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, y=y):
    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    for row in y_pred:
        max_index = np.argmax(row[indices_to_transform])
        row[indices_to_transform] = 0
        row[indices_to_transform[max_index]] = 1

    mse = mean_squared_error(y_val, y_pred)
    total_mse.append(mse)

t2 = time.time()


print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
print(t2 - t1)