import xgboost as xgb
import numpy as np
import pandas as pd
from transformers import AutoImageProcessor, ViTMSNModel
import torch
import torch.nn as nn
from preprocessing import  preprocessing


# Custom loss function (squared loss for simplicity)
def custom_loss(preds, dtrain):
    labels = dtrain.get_label()
    nested_labels = [labels[i:i + 8] for i in range(0, len(labels), 8)]

    diff = preds - nested_labels

    weights = np.array([1, 1, 1, 4, 5, 6, 7, 8])  # Placeholder for custom weights

    weighted_diff = diff * weights
    # Assuming weights are based on some logic related to labels or features
    grad = 2 * weighted_diff # Gradient
    hess = 2 * np.ones_like(preds) * weights  # Hessian

    return grad.flatten(), hess.flatten()


processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

df = preprocessing()

hidden_representations = []
labels = []

pool = nn.MaxPool2d(2, 2)

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


# Example loading data
# X_train, y_train should be your data and labels respectively

X_train, X_val = X.iloc[0:61], X.iloc[61:]
y_train, y_val = y.iloc[0:61], y.iloc[61:]

dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy())

params = {
    'max_depth': 3,
    'eta': 0.1,
    'silent': 1,
    'objective': 'reg:squarederror'
}

model = xgb.train(params, dtrain, num_boost_round=10, obj=custom_loss)
