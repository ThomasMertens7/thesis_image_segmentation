from preprocessing import preprocessing
from transformers import ViTMSNModel, AutoImageProcessor
import torch
import xgboost as xgb
import pandas as pd


def renew_model():
    processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
    model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

    df = preprocessing().sample(frac=1).reset_index(drop=True)

    hidden_representations = []
    labels = []
    widths = []
    heights = []

    for i, row in df.iterrows():
        inputs = processor(images=row['image'], return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        pooled_hidden_state = last_hidden_states
        widths.append(row['image'].size[0])
        heights.append(row['image'].size[1])

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
    X["width"] = widths
    X["height"] = heights

    y = pd.DataFrame(labels, columns=['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                             'lambda', 'inner_iterations', 'outer_iterations', 'num_points'])

    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train = X
    y_train = y

    model.fit(X_train, y_train)

    model.save_model('model.bin')

renew_model()