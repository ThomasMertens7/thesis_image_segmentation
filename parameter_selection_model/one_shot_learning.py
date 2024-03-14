from transformers import AutoImageProcessor, ViTMSNModel
import torch
from PIL import Image
from preprocessing import  preprocessing_with_animal
import numpy as np


def euclidean_distance(matrix1, matrix2):
    # Compute the squared differences element-wise
    squared_diff = (matrix1 - matrix2) ** 2

    # Sum the squared differences along the specified axes
    squared_diff_sum = squared_diff.sum()  # Sum along the last two axes

    # Take the square root to get the Euclidean distance
    euclidean_dist = np.sqrt(squared_diff_sum)

    return euclidean_dist


image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing_with_animal()

hidden_representations = []
for i, row in df.iterrows():
    inputs = image_processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    hidden_representations.append(last_hidden_states)

df['image'] = hidden_representations
print(df.head)

train_data = df.iloc[:60]
test_data = df.iloc[61:69]

for i1, row1 in test_data.iterrows():
    min_distance = float('inf')
    best_index = None
    best_animal = None
    best_parameters = None

    for i2, row2 in train_data.iterrows():
        curr_distance = euclidean_distance(row1["image"][0], row2["image"][0])

        if curr_distance < min_distance:
            min_distance = curr_distance
            best_index = i2
            best_animal = row2['animal']
            best_parameters = row2[['edge_indicator', 'alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']]

    print("For index 1: " + row1['animal'] + ", it matches best for parameters" + str(best_parameters))






