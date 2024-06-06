from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, ViTMSNModel, ResNetModel, ViTModel, ViTMAEModel
import torch
from preprocessing import preprocessing, get_mean_and_var, normalize_list, preprocessing_newer, get_groups
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
import time

t1 = time.time()

processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

df = preprocessing_newer()

widths = []
heights = []
hidden_representations = []
labels = []

for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    flattened_tensor = last_hidden_states.view(-1)

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(hidden_representations_lists)

pca = PCA(n_components=151)

X_pca = pca.fit_transform(X_scaled)

X = pd.DataFrame(X_pca)

X["width"] = widths
X["heights"] = heights

groups = get_groups(df)

y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

### n_estimators = 100, learning_rate = 0.3, max_depth = 6
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.3, max_depth=9)
kf = GroupKFold(n_splits=10)

total_mae = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, groups=groups):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    total_mae.append(mae)

    for row in y_pred:
        max_index = np.argmax(row[indices_to_transform])
        row[indices_to_transform] = 0
        row[indices_to_transform[max_index]] = 1

print(sum(total_mae) / len(total_mae))
print(np.var(total_mae))

t2 = time.time()

print(t2 - t1)