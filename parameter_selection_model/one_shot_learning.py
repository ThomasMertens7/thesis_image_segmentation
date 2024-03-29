from sklearn.model_selection import KFold
from transformers import AutoImageProcessor, ViTMSNModel
import torch
from preprocessing import preprocessing_with_animal
import numpy as np
from sklearn.metrics import mean_squared_error
import time

t1 = time.time()


def euclidean_distance(matrix1, matrix2):
    # Compute the squared differences element-wise
    squared_diff = (matrix1 - matrix2) ** 2

    # Sum the squared differences along the specified axes
    squared_diff_sum = squared_diff.sum()  # Sum along the last two axes

    # Take the square root to get the Euclidean distance
    euclidean_dist = np.sqrt(squared_diff_sum)

    return euclidean_dist


image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-base")

df = preprocessing_with_animal()

hidden_representations = []
for i, row in df.iterrows():
    inputs = image_processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    hidden_representations.append(last_hidden_states)

df['image'] = hidden_representations

kf = KFold(n_splits=10, shuffle=True)
total_mse = list()

for train_index, val_index in kf.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[val_index]

    for i1, row1 in test_data.iterrows():
        min_distance = float('inf')
        best_index = None
        best_animal = None
        predicted_parameters = None

        for i2, row2 in train_data.iterrows():
            curr_distance = euclidean_distance(row1["image"][0], row2["image"][0])

            if curr_distance < min_distance:
                min_distance = curr_distance
                best_index = i2
                best_animal = row2['animal']
                predicted_parameters = row2[['edge_indicator', 'alpha', 'sigma', 'lambda', 'inner_iterations',
                                             'outer_iterations']]

        predicted_parameters["GEODESIC_DISTANCE"] = 1 if \
            (predicted_parameters['edge_indicator'] == "EdgeIndicator.GEODESIC_DISTANCE") else 0
        predicted_parameters["EUCLIDEAN_DISTANCE"] = 1 if \
            (predicted_parameters['edge_indicator'] == "EdgeIndicator.EUCLIDEAN_DISTANCE") else 0
        predicted_parameters["SCALAR_DIFFERENCE"] = 1 if \
            (predicted_parameters['edge_indicator'] == "EdgeIndicator.SCALAR_DIFFERENCE") else 0
        predicted_parameters = predicted_parameters[
            ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
             'sigma', 'lambda', 'outer_iterations', 'inner_iterations']]

        row1["GEODESIC_DISTANCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.GEODESIC_DISTANCE") else 0
        row1["EUCLIDEAN_DISTANCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.EUCLIDEAN_DISTANCE") else 0
        row1["SCALAR_DIFFERENCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.SCALAR_DIFFERENCE") else 0
        actual_parameters = row1[
            ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
             'sigma', 'lambda', 'outer_iterations', 'inner_iterations']]
        mse = mean_squared_error(predicted_parameters, actual_parameters)
        total_mse.append(mse)

t2 = time.time()

print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
print(t2-t1)





