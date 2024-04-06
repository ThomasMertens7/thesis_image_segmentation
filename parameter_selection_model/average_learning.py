from sklearn.model_selection import GroupKFold
from transformers import AutoImageProcessor, ViTMSNModel
import torch
from preprocessing import preprocessing_with_animal, get_mean_and_var, normalize_list, preprocessing_newer, get_groups, preprocessing
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


df = preprocessing_newer()

groups = get_groups(df)

print(groups)

kf = GroupKFold(n_splits=10)
total_mse = list()

for train_index, val_index in kf.split(df, groups=groups):
    print(train_index, val_index)
    train_data = df.iloc[train_index]
    test_data = df.iloc[val_index]

    predicted_parameters = train_data.mean()

    predicted_parameters = predicted_parameters[['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE',
                                                 'alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations',
                                                 'num_points']]

    #max_key = max(['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE'],
    #              key=lambda k: predicted_parameters[k])

    # Set the key with the maximum value to 1, and the others to 0
    #for key in ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE']:
    #    predicted_parameters[key] = 1 if key == max_key else 0

    for i1, row1 in test_data.iterrows():

        actual_parameters = row1[
            ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
             'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']]

        mse = mean_squared_error(predicted_parameters.tolist(), list(actual_parameters))
        abs = mean_absolute_error(predicted_parameters.tolist(), list(actual_parameters))
        print("new results")
        print(predicted_parameters.tolist())
        print(list(actual_parameters))
        print(abs)
    total_mse.append(np.mean(abs))

t2 = time.time()

print(sum(total_mse) / len(total_mse))
print(total_mse)
print(np.var(total_mse))
print(t2-t1)





