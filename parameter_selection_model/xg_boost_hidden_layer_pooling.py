from transformers import AutoImageProcessor, ViTMSNModel, ResNetModel, ViTModel, ViTMAEModel, ImageGPTConfig, \
    ImageGPTModel
import torch
from preprocessing import preprocessing
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
import time

t1 = time.time()

processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

df = preprocessing()

hidden_representations = []
labels = []

pool = nn.MaxPool2d(2, 2)

for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    pooled_hidden_state = last_hidden_states

    flattened_tensor = pooled_hidden_state.view(-1)

    hidden_representations.append(flattened_tensor)

    labels.append([
        row['SCALAR_DIFFERENCE'],
        row['EUCLIDEAN_DISTANCE'],
        row['GEODESIC_DISTANCE'],
        row['alpha'],
        row['sigma'],
        row['lambda'],
        row['inner_iterations'],
        row['outer_iterations']
    ])

hidden_representations_lists = [tensor.tolist() for tensor in hidden_representations]
X = pd.DataFrame(hidden_representations_lists)

y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations'])

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
    print("finished split")

    mse = mean_squared_error(y_val, y_pred)
    total_mse.append(mse)

print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
t2 = time.time()
print(t2 - t1)