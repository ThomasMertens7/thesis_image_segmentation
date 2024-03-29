import numpy as np
import math

from scipy.ndimage import distance_transform_edt


def get_distance_contour(segmentation_matrix, row_index, column_index):
    if segmentation_matrix[row_index][column_index] == 0:
        segmentation_matrix = 1 - segmentation_matrix

    distance_matrix = distance_transform_edt(segmentation_matrix)
    distance = distance_matrix[row_index][column_index]

    return distance


def perceptual_difference_unit(npy_file_name, calculated_contour_name):
    ground_segmenation_matrix = np.load(npy_file_name)
    found_segmentation_matrix = np.load(calculated_contour_name)

    result = 0
    D = math.sqrt(math.pow(len(found_segmentation_matrix), 2) + math.pow(len(found_segmentation_matrix[0]), 2))

    for row_index in range(len(found_segmentation_matrix)):
        for column_index in range(len(found_segmentation_matrix[0])):
            if found_segmentation_matrix[row_index][column_index] == 1 and\
                    ground_segmenation_matrix[row_index][column_index] == 0:
                result += (math.log(1 + get_distance_contour(ground_segmenation_matrix, row_index, column_index)) / D)
            elif found_segmentation_matrix[row_index][column_index] == 0 and \
                    ground_segmenation_matrix[row_index][column_index] == 1:
                result += (get_distance_contour(ground_segmenation_matrix, row_index, column_index) / D)
    total_elements = found_segmentation_matrix.shape[0] * found_segmentation_matrix.shape[1]

    precision = 1 - (result / total_elements)
    return precision

