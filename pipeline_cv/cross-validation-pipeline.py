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

kf = GroupKFold(n_splits=3)

total_prec = list()
total_diff = list()

for train_index, val_index in kf.split(X, y=y, groups=groups):
    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
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