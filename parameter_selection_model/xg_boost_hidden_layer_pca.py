from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, ViTMSNModel, ResNetModel, ViTModel
from diffusers import AutoencoderKL
import torch
from preprocessing import preprocessing, get_mean_and_var, normalize_list, preprocessing_newer, get_groups
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
import time

t1 = time.time()

processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

df = preprocessing_newer()

hidden_representations = []
labels = []

for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    flattened_tensor = last_hidden_states.view(-1)

    hidden_representations.append(flattened_tensor)

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

pca = PCA(n_components=67)

X_pca = pca.fit_transform(X_scaled)

X = pd.DataFrame(X_pca)

groups = get_groups(df)

y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

model = xgb.XGBRegressor(objective='reg:squarederror')
kf = GroupKFold(n_splits=10)

total_mse = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, y=y, groups=groups):
    print(train_index)
    print(val_index)
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    for row in y_pred:
        max_index = np.argmax(row[indices_to_transform])
        row[indices_to_transform] = 0
        row[indices_to_transform[max_index]] = 1

    mse = mean_absolute_error(y_val, y_pred)
    total_mse.append(mse)

    total_mse.append(mse)

print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))

t2 = time.time()

print(t2 - t1)