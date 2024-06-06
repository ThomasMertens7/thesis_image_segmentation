from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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

df = df.sample(frac=1, random_state=1)

groups = get_groups(df)


kf = GroupKFold(n_splits=10)

total_mae = list()

for train_index, val_index in kf.split(df, groups=groups):
    train_data = df.iloc[train_index]
    test_data = df.iloc[val_index]

    predicted_parameters = train_data.mean()

    predicted_parameters = predicted_parameters[['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE',
                                                 'alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations',
                                                 'num_points']]

    abs_list = []
    for i1, row1 in test_data.iterrows():

        actual_parameters = row1[
            ['GEODESIC_DISTANCE', 'EUCLIDEAN_DISTANCE', 'SCALAR_DIFFERENCE', 'alpha',
             'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']]

        mae = mean_absolute_error(list(actual_parameters), predicted_parameters.tolist())
        abs_list.append(mae)

    total_mae.append(np.mean(abs_list))

t2 = time.time()

print(total_mae)
print(sum(total_mae) / len(total_mae))
print(np.var(total_mae))





