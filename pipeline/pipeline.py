from PIL import Image
from perceptual_difference_unit import perceptual_difference_unit
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from execute_example import execute
from get_optimal_parameters import get_optimal_parameters
from get_ground_truth_segmentation import get_ground_truth_segmentation
from openpyxl import load_workbook
import gc
import numpy as np
import pandas as pd


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
    image_path = '../database/all_imgs/' + image_name + '.jpeg'
    image = Image.open(image_path)

    print(1)
    bounding_box, iter_inner, iter_outer, lmbda, alpha, sigma, edge_indicator, num_points \
        = get_optimal_parameters(image, image_name)
    gc.collect()

    print(2)

    execute(image_path, bounding_box, iter_inner, iter_outer, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
            edge_indicator, num_points)
    gc.collect()

    print(3)

    get_ground_truth_segmentation(image_name)
    gc.collect()

    print(4)
    precision = perceptual_difference_unit('ground_truth.npy', 'prediction.npy')

    print(5)

    animal, level = get_animal_level(image_name, precision)

    print(6)
    new_row_data = (image_name, str(bounding_box), str(edge_indicator), alpha, sigma, lmbda, iter_inner, iter_outer,
                    num_points, precision, animal, level)

    array = np.load('ground_truth.npy')
    df = pd.DataFrame(array)

    df.to_excel('output.xlsx', index=False)

    print(7)
    workbook = load_workbook('latest_data.xlsx')
    sheet = workbook['data']
    sheet.append(new_row_data)
    workbook.save("latest_data.xlsx")
    print(8)


segment("horse-53")










