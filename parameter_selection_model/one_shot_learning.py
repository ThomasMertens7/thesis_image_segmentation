from sklearn.model_selection import KFold
from transformers import AutoImageProcessor, ViTMSNModel
import torch
from preprocessing import preprocessing_with_animal, get_mean_and_var, normalize_list
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import heapq

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
    train_data = df.iloc[train_index].reset_index(drop=True)
    test_data = df.iloc[val_index]

    for i1, row1 in test_data.iterrows():
        max_heap = []
        index = 0

        for i2, row2 in train_data.iterrows():
            try:
                curr_sim = 1/euclidean_distance(row1["image"][0], row2["image"][0]).item()
            except Exception:
                curr_sim = 100000

            heapq.heappush(max_heap, (curr_sim, i2))

            if len(max_heap) > 5:
                heapq.heappop(max_heap)

        for index in range(len(max_heap)):
            curr_labels = train_data[['edge_indicator', 'alpha', 'sigma', 'lambda', 'inner_iterations',
                                                  'outer_iterations']].iloc[max_heap[index][1]]

            curr_labels["GEODESIC_DISTANCE"] = 1 if \
                (curr_labels['edge_indicator'] == "EdgeIndicator.GEODESIC_DISTANCE") else 0
            curr_labels["EUCLIDEAN_DISTANCE"] = 1 if \
                (curr_labels['edge_indicator'] == "EdgeIndicator.EUCLIDEAN_DISTANCE") else 0
            curr_labels["SCALAR_DIFFERENCE"] = 1 if \
                (curr_labels['edge_indicator'] == "EdgeIndicator.SCALAR_DIFFERENCE") else 0
            curr_labels = curr_labels[['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
                            'sigma', 'lambda', 'inner_iterations', 'outer_iterations']]

            max_heap[index] = (max_heap[index][0], curr_labels)

        ## Parameters
        predicted_parameters = {
            'GEODESIC_DISTANCE': 0,
            'EUCLIDEAN_DISTANCE': 0,
            'SCALAR_DIFFERENCE': 0,
            'alpha': 0,
            'sigma': 0,
            'lambda': 0,
            'inner_iterations': 0,
            'outer_iterations': 0
        }

        total_sim = sum([elem[0] for elem in max_heap])

        for current_result in max_heap:
            current_parameters = dict(current_result[1])
            current_sim = current_result[0]
            predicted_parameters = {key: predicted_parameters[key] +
                current_parameters[key] * current_sim/total_sim for key in predicted_parameters}

        max_key = max(['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE'],
                      key=lambda k: predicted_parameters[k])

        # Set the key with the maximum value to 1, and the others to 0
        for key in ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE']:
            predicted_parameters[key] = 1 if key == max_key else 0

        row1["GEODESIC_DISTANCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.GEODESIC_DISTANCE") else 0
        row1["EUCLIDEAN_DISTANCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.EUCLIDEAN_DISTANCE") else 0
        row1["SCALAR_DIFFERENCE"] = 1 if (row1['edge_indicator'] == "EdgeIndicator.SCALAR_DIFFERENCE") else 0
        actual_parameters = row1[
            ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
             'sigma', 'lambda', 'inner_iterations', 'outer_iterations']]

        mse = mean_squared_error(list(predicted_parameters.values()), list(actual_parameters))
        total_mse.append(mse)

t2 = time.time()

print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
print(t2-t1)





