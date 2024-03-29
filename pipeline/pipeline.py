from PIL import Image
from perceptual_difference_unit import perceptual_difference_unit
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction
from execute_example import execute
from get_optimal_parameters import get_optimal_parameters
from get_ground_truth_segmentation import get_ground_truth_segmentation
from openpyxl import load_workbook
import gc


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

    new_row_data = (image_name, str(bounding_box), str(edge_indicator), alpha, sigma, lmbda, iter_inner, iter_outer, precision)
    workbook = load_workbook('latest_data.xlsx')
    sheet = workbook['data']
    sheet.append(new_row_data)
    workbook.save("latest_data.xlsx")

segment("dog-52")









