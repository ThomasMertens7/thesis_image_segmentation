from transformers import AutoImageProcessor, ViTMSNModel, ResNetModel, ViTModel, ViTMAEModel, ImageGPTConfig, \
    ImageGPTModel
import torch
from preprocessing import preprocessing, get_mean_and_var, normalize_list, preprocessing_newer, get_groups
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
import torch.nn as nn
import time


t1 = time.time()
processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing_newer()

hidden_representations = []
widths = []
heights = []
labels = []

for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.size)
    pooled_hidden_state = last_hidden_states

    flattened_tensor = pooled_hidden_state.view(-1)

    hidden_representations.append(flattened_tensor)
    widths.append(row['image'].size[0])
    heights.append(row['image'].size[1])

    labels.append([
        row['SCALAR_DIFFERENCE'],
        row['EUCLIDEAN_DISTANCE'],
        row['GEODESIC_DISTANCE'],
        row['alpha'],
        row['sigma'],
        row['lambda'],
        row['inner_iterations'],
        row['outer_iterations'],
        row['num_points']
    ])

hidden_representations_lists = [tensor.tolist() for tensor in hidden_representations]
X = pd.DataFrame(hidden_representations_lists)
X["width"] = widths
X["heights"] = heights

groups = get_groups(df)

y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

kf = GroupKFold(n_splits=10)

total_mse = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, y=y, groups=groups):
    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_absolute_error(y_val, y_pred)
    print(mse)
    total_mse.append(mse)

    for row in y_pred:
        max_index = np.argmax(row[indices_to_transform])
        row[indices_to_transform] = 0
        row[indices_to_transform[max_index]] = 1
    print("finished split")



print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
print(total_mse)
t2 = time.time()
print(t2 - t1)