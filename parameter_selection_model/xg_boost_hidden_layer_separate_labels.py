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
sv = y['SCALAR_DIFFERENCE']
ga = y['GEODESIC_DISTANCE']
ea = y['EUCLIDEAN_DISTANCE']
lmbda = y['lambda']
alpha = y['alpha']
sigma = y['sigma']
inner_iter = y['inner_iterations']
outer_iter = y['outer_iterations']
ap = y['num_points']


kf = GroupKFold(n_splits=10)

total_mae = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(X, y=y, groups=groups):
    model_P_SV = xgb.XGBRegressor(objective='reg:squarederror')
    model_P_GA = xgb.XGBRegressor(objective='reg:squarederror')
    model_P_EA = xgb.XGBRegressor(objective='reg:squarederror')
    model_lambda = xgb.XGBRegressor(objective='reg:squarederror')
    model_alpha = xgb.XGBRegressor(objective='reg:squarederror')
    model_sigma = xgb.XGBRegressor(objective='reg:squarederror')
    model_inner_iter = xgb.XGBRegressor(objective='reg:squarederror')
    model_outer_iter = xgb.XGBRegressor(objective='reg:squarederror')
    model_ap = xgb.XGBRegressor(objective='reg:squarederror')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    P_SV_train, P_SV_val = sv.iloc[train_index], sv.iloc[val_index]
    P_GA_train, P_GA_val = ga.iloc[train_index], ga.iloc[val_index]
    P_EA_train, P_EA_val = ea.iloc[train_index], ea.iloc[val_index]
    lmbda_train, lmbda_val = lmbda.iloc[train_index], lmbda.iloc[val_index]
    alpha_train, alpha_val = alpha.iloc[train_index], alpha.iloc[val_index]
    sigma_train, sigma_val = sigma.iloc[train_index], sigma.iloc[val_index]
    inner_iter_train, inner_iter_val = inner_iter.iloc[train_index], inner_iter.iloc[val_index]
    outer_iter_train, outer_iter_val = outer_iter.iloc[train_index], outer_iter.iloc[val_index]
    ap_train, ap_val = ap.iloc[train_index], ap.iloc[val_index]

    model_P_SV.fit(X_train, P_SV_train)
    model_P_GA.fit(X_train, P_GA_train)
    model_P_EA.fit(X_train, P_EA_train)
    model_lambda.fit(X_train, lmbda_train)
    model_alpha.fit(X_train, alpha_train)
    model_sigma.fit(X_train, sigma_train)
    model_inner_iter.fit(X_train, inner_iter_train)
    model_outer_iter.fit(X_train, outer_iter_train)
    model_ap.fit(X_train, ap_train)

    P_SV_pred = model_P_SV.predict(X_val)
    P_GA_pred = model_P_GA.predict(X_val)
    P_EA_pred = model_P_EA.predict(X_val)
    lambda_pred = model_lambda.predict(X_val)
    alpha_pred = model_alpha.predict(X_val)
    sigma_pred = model_sigma.predict(X_val)
    inner_iter_pred = model_inner_iter.predict(X_val)
    outer_iter_pred = model_outer_iter.predict(X_val)
    ap_pred = model_ap.predict(X_val)

    mae = (mean_absolute_error(P_SV_val, P_SV_pred) + mean_absolute_error(P_GA_val, P_GA_pred) +
          mean_absolute_error(P_EA_val, P_EA_pred) + mean_absolute_error(lmbda_val, lambda_pred) +
          mean_absolute_error(alpha_val, alpha_pred) + mean_absolute_error(sigma_val, sigma_pred) +
          mean_absolute_error(inner_iter_val, inner_iter_pred) + mean_absolute_error(outer_iter_val, outer_iter_pred) +
          mean_absolute_error(ap_val, ap_pred)) / 9

    total_mae.append(mae)
    print(mae)
    print("finished split")



print(sum(total_mae) / len(total_mae))
print(np.var(total_mae))
print(total_mae)
t2 = time.time()
print(t2 - t1)