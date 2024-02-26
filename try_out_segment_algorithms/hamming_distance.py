import numpy as np


def hamming_distance(npy_file_name, calculated_contour_name):
    ground_segmenation_matrix = np.load(npy_file_name)
    found_segmentation_matrix = np.load(calculated_contour_name)

    result = 0

    for row_index in range(len(found_segmentation_matrix)):
        for column_index in range(len(found_segmentation_matrix[0])):
            if found_segmentation_matrix[row_index][column_index] != ground_segmenation_matrix[row_index][column_index]:
                result += 1

    total_elements = found_segmentation_matrix.shape[0] * found_segmentation_matrix.shape[1]

    precision = 1 - (result / total_elements)
    return precision







