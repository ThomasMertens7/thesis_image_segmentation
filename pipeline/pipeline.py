from PIL import Image
from perceptual_difference_unit import perceptual_difference_unit
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from segment_example import execute
from get_optimal_parameters import get_optimal_parameters
from get_ground_truth_segmentation import get_ground_truth_segmentation
from openpyxl import load_workbook
import gc
import numpy as np
import pandas as pd
import os



def get_animal_level(image_name, precision):
    image_parts = image_name.split("-")
    animal = image_parts[0]

    level = None
    if animal == "dog":
        level = "Easy" if precision >= 0.99992 else "Hard"
    elif animal == "cow":
        level = "Easy" if precision >= 0.99987 else "Hard"
    elif animal == "sheep":
        level = "Easy" if precision >= 0.99988 else "Hard"
    elif animal == "horse":
        level = "Easy" if precision >= 0.9998 else "Hard"
    return animal, level


def segment(image_name):
    image_path = '../database/all_imgs_new/' + image_name + '.jpeg'
    image = Image.open(image_path)

    bounding_box, iter_inner, iter_outer, lmbda, alpha, sigma, edge_indicator, num_points \
        = get_optimal_parameters(image, image_name)
    gc.collect()

    execute(image_path, bounding_box, iter_inner, iter_outer, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
            edge_indicator, num_points)
    gc.collect()

    get_ground_truth_segmentation(image_name)
    gc.collect()

    precision = perceptual_difference_unit('ground_truth.npy', 'prediction.npy')
    print("The precision is " + str(precision))

    animal, level = get_animal_level(image_name, precision)

    new_row_data = (image_name, str(bounding_box), str(edge_indicator), alpha, sigma, lmbda, iter_inner, iter_outer,
                    num_points, precision, animal, level)

    workbook = load_workbook('latest_data.xlsx')
    sheet = workbook['data']
    sheet.append(new_row_data)
    workbook.save("latest_data.xlsx")

    os.remove('prediction.npy')
    os.remove('ground_truth.npy')


for batch in range(10):
    for index in range(1, 6):
        overall_index = str(batch * 5 + index)
        print("The overall index is " + str(overall_index))
        segment('dog-new-' + overall_index)
        segment('cow-new-' + overall_index)
        segment('sheep-new-' + overall_index)
        segment('horse-new-' + overall_index)










