from PIL import Image
from perceptual_difference_unit import perceptual_difference_unit
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from execute_example import execute
from get_optimal_parameters import get_optimal_parameters
from get_ground_truth_segmentation import get_ground_truth_segmentation
import gc
import pandas as pd
import numpy as np


def segment(image_name):
    image_path = '../database/all_imgs/' + image_name + '.jpeg'
    image = Image.open(image_path)

    bounding_box, iter_inner, iter_outer, lmbda, alpha, sigma, edge_indicator = get_optimal_parameters(image, image_name)
    gc.collect()

    execute(image_path, bounding_box, iter_inner, iter_outer, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL, edge_indicator, 100)
    gc.collect()

    get_ground_truth_segmentation(image_name)
    gc.collect()

    precision = perceptual_difference_unit('ground_truth.npy', 'prediction.npy')

    return precision


precision_data = list()
df = pd.read_excel('old_data.xlsx')

for image_name in df['image'].unique():
    new_prec = segment(image_name)
    print(new_prec)
    precision_data.append(new_prec)

print(precision_data)
print(np.mean(precision_data))
print(np.var(precision_data))