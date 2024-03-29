from pipeline.preprocessing import preprocessing
from transformers import ViTMSNModel, AutoImageProcessor
import torch
import xgboost as xgb
import pandas as pd

def renew_model():
    processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
    model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

    df = preprocessing()

    hidden_representations = []
    labels = []

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

    model = xgb.XGBRegressor(objective='reg:squarederror')

    X_train = X
    y_train = y

    model.fit(X_train, y_train)

    model.save_model('model.bin')

renew_model()