import pandas as pd
from safetensors import torch
from transformers import AutoImageProcessor, ViTMSNModel
import xgboost as xgb
import torch
from geomeansegmentation.image_segmentation.drlse_segmentation import EdgeIndicator
from parameter_selection_model.preprocessing import preprocessing, preprocessing_filter
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour
import numpy as np
import gc


def check_string_in_excel_column(file_path, column_name, string_to_check):
    df = pd.read_excel(file_path, engine='openpyxl')

    # Check if the string is in the specified column
    if string_to_check in df[column_name].values:
        return True
    else:
        return False


def get_optimal_parameters(image, image_name):
    processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
    msn_model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

    model = None
    if check_string_in_excel_column("latest_data.xlsx", "image", image_name):
        df = preprocessing_filter(image_name)

        hidden_representations = []
        labels = []

        for i, row in df.iterrows():
            inputs = processor(images=row['image'], return_tensors="pt")

            with torch.no_grad():
                outputs = msn_model(**inputs)

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
                row['outer_iterations'],
                row['num_points']
            ])

        hidden_representations_lists = [tensor.tolist() for tensor in hidden_representations]
        X = pd.DataFrame(hidden_representations_lists)
        y = pd.DataFrame(labels,
                             columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                                      'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

        model = xgb.XGBRegressor(objective='reg:squarederror')

        X_train = X
        y_train = y

        model.fit(X_train, y_train)

    else:
        model = xgb.XGBRegressor()
        model.load_model("model.bin")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = msn_model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    flattened_tensor = last_hidden_state.view(-1)
    X = pd.DataFrame([flattened_tensor.tolist()])

    del msn_model
    gc.collect()

    image_parameters = model.predict(X)[0]

    del model
    gc.collect()

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

