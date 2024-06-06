from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTMSNModel
from perceptual_difference_unit import perceptual_difference_unit
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from execute_example import execute
from get_optimal_parameters import get_optimal_parameters, get_bounding_box
from get_ground_truth_segmentation import get_ground_truth_segmentation
from openpyxl import load_workbook
from preprocessing import preprocessing_cv
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import EdgeIndicator
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb

max_min = {"alpha": (1.75, 9.5), "sigma": (0, 6.875), "lambda": (-0.5, 6.875), "inner_iter": (11, 29),
           "outer_iter": (30, 58), "num_points": (50, 162)}


def get_groups(df):
    all_names = []
    all_indexes = []

    index = -1
    for i, row in df.iterrows():
        if row['name'] not in all_names:
            index += 1
            all_names.append(row['name'])
            all_indexes.append(index)

        elif row['name'] in all_names:
            new_index = all_names.index(row['name'])
            all_indexes.append(new_index)

    return np.array(all_indexes)


def segment(y_pred, image_name):
    image_path = '../database/all_imgs/' + image_name + '.jpeg'

    bounding_box = get_bounding_box(image_name)

    edge_indicators = y_pred[0:3]
    max_index = np.argmax(edge_indicators)
    edge_indicator = None
    if max_index == 0:
        edge_indicator = EdgeIndicator.SCALAR_DIFFERENCE
    elif max_index == 1:
        edge_indicator = EdgeIndicator.EUCLIDEAN_DISTANCE
    elif max_index == 2:
        edge_indicator = EdgeIndicator.GEODESIC_DISTANCE

    print(
            image_path,
            bounding_box,
            int(y_pred[6] * (max_min["inner_iter"][1] - max_min["inner_iter"][0]) + max_min["inner_iter"][0]),
            int(y_pred[7] * (max_min["outer_iter"][1] - max_min["outer_iter"][0]) + max_min["outer_iter"][0]),
            y_pred[5] * (max_min["lambda"][1] - max_min["lambda"][0]) + max_min["lambda"][0],
            y_pred[3] * (max_min["alpha"][1] - max_min["alpha"][0]) + max_min["alpha"][0],
            1.5,
            y_pred[4] * (max_min["sigma"][1] - max_min["sigma"][0]) + max_min["sigma"][0],
            PotentialFunction.DOUBLE_WELL,
            edge_indicator,
            int(y_pred[8] * (max_min["num_points"][1] - max_min["num_points"][0]) + max_min["num_points"][0])
    )

    execute(image_path, bounding_box,
            int(y_pred[6] * (max_min["inner_iter"][1] - max_min["inner_iter"][0]) + max_min["inner_iter"][0]),
            int(y_pred[7] * (max_min["outer_iter"][1] - max_min["outer_iter"][0]) + max_min["outer_iter"][0]),
            y_pred[5] * (max_min["lambda"][1] - max_min["lambda"][0]) + max_min["lambda"][0],
            y_pred[3] * (max_min["alpha"][1] - max_min["alpha"][0]) + max_min["alpha"][0],
            1.5,
            y_pred[4] * (max_min["sigma"][1] - max_min["sigma"][0]) + max_min["sigma"][0],
            PotentialFunction.DOUBLE_WELL,
            edge_indicator,
            int(y_pred[8] * (max_min["num_points"][1] - max_min["num_points"][0]) + max_min["num_points"][0]))
    gc.collect()

    precision = perceptual_difference_unit('../try_out_segment_algorithms/' + image_name + ".npy", 'prediction.npy')
    print("The precision is " + str(precision))

    os.remove('prediction.npy')

    return precision


processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing_cv().sample(frac=1, random_state=42).reset_index(drop=True)

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
X["height"] = heights

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

kf = GroupKFold(n_splits=3)

total_prec = list()
total_diff = list()

for train_index, val_index in kf.split(X, y=y, groups=groups):
    model = xgb.XGBRegressor(objective='reg:squarederror')
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

    y_pred = np.concatenate((P_SV_pred[:, np.newaxis], P_GA_pred[:, np.newaxis], P_EA_pred[:, np.newaxis],
                             alpha_pred[:, np.newaxis], sigma_pred[:, np.newaxis], lambda_pred[:, np.newaxis],
                             inner_iter_pred[:, np.newaxis], outer_iter_pred[:, np.newaxis], ap_pred[:, np.newaxis]),
                            axis=1)
    names = df.iloc[val_index]['name']

    cv_prec = list()
    i = 0
    for index in val_index:
        try:
            new_prec = segment(y_pred[i], df['name'].iloc[index])
            cv_prec.append(new_prec)
            total_diff.append((df['name'].iloc[index], new_prec))

        except Exception as e:
            print(df['name'].iloc[index] + " failed...")
            print(e)
        i += 1

    total_prec.append(np.mean(cv_prec))
    print("finished split")

print(total_diff)
print(sum(total_prec) / len(total_prec))
print(np.var(total_prec))