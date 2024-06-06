from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTMSNModel
from perceptual_difference_unit import perceptual_difference_unit
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from execute_example import execute
from get_optimal_parameters import get_optimal_parameters, get_bounding_box
from get_ground_truth_segmentation import get_ground_truth_segmentation
from openpyxl import load_workbook
from preprocessing import preprocessing_cv_avg
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import EdgeIndicator
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb


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

    print(image_path, bounding_box,
            int(y_pred[6]),
            int(y_pred[7]),
            y_pred[5],
            y_pred[3],
            1.5,
            y_pred[4],
            PotentialFunction.DOUBLE_WELL,
            edge_indicator,
            int(y_pred[8]))

    execute(image_path, bounding_box,
            int(y_pred[6]),
            int(y_pred[7]),
            y_pred[5],
            y_pred[3],
            1.5,
            y_pred[4],
            PotentialFunction.DOUBLE_WELL,
            edge_indicator,
            int(y_pred[8])
            )
    gc.collect()

    precision = perceptual_difference_unit('../try_out_segment_algorithms/' + image_name + ".npy", 'prediction.npy')
    print("The precision is " + str(precision))

    os.remove('prediction.npy')

    return precision


processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing_cv_avg().sample(frac=1, random_state=42).reset_index(drop=True)

labels = []

for i, row in df.iterrows():
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

groups = get_groups(df)

y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

kf = GroupKFold(n_splits=3)

total_prec = list()
total_diff = list()
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(y, groups=groups):

    y_train = y.iloc[train_index]

    y_pred = y_train.mean().tolist()

    names = df.iloc[val_index]['name']

    cv_prec = list()
    i = 0
    for index in val_index:
        try:
            new_prec = segment(y_pred, df['name'].iloc[index])
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