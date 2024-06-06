import pandas as pd
from safetensors import torch
from transformers import AutoImageProcessor, ViTMSNModel
import xgboost as xgb
import torch
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import EdgeIndicator
from parameter_selection_model.preprocessing import preprocessing, preprocessing_filter
from RAMON_segmentinitialcontour.segment_initial_countour_train import get_initial_contour
import numpy as np
import gc


def check_string_in_excel_column(file_path, column_name, string_to_check):
    df = pd.read_excel(file_path, engine='openpyxl')

    # Check if the string is in the specified column
    if string_to_check in df[column_name].values:
        return True
    else:
        return False


def get_bounding_box(image_name):
    image_path = '../database/all_imgs/' + image_name + '.jpeg'
    bounding_box = get_initial_contour(image_path)
    return bounding_box


def get_optimal_parameters(image_name, hidden_layer, model):
    image_parameters = model.predict(hidden_layer)[0]

    image_path = '../database/all_imgs/' + image_name + '.jpeg'
    bounding_box = get_initial_contour(image_path)

    edge_indicators = image_parameters[0:3]
    max_index = np.argmax(edge_indicators)
    edge_indicator = None
    if max_index == 0:
        edge_indicator = EdgeIndicator.SCALAR_DIFFERENCE
    elif max_index == 1:
        edge_indicator = EdgeIndicator.EUCLIDEAN_DISTANCE
    elif max_index == 2:
        edge_indicator = EdgeIndicator.GEODESIC_DISTANCE
    alpha = image_parameters[3]
    sigma = image_parameters[4]
    lmbda = image_parameters[5]
    iter_inner = image_parameters[6]
    iter_outer = image_parameters[7]
    num_points = image_parameters[8]

    return bounding_box, int(iter_inner), int(iter_outer), lmbda, alpha, sigma, edge_indicator, int(num_points)

