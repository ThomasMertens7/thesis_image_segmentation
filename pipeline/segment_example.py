import json

import numpy as np
import cv2
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import perform_segmentation, PotentialFunction, EdgeIndicator, construct_g
from figure.show_figure import show_all, show_lsf, show_contour
from segmentrecognition.image_recognition_recognize import recognize_animal
from RAMON_examples.convert_contour import convert_to_0_1
def execute(img_path, initial_countour_coordinates, iter_inner, iter_outer, lmbda, alfa, epsilon, sigma, potential_function, edge_indicator, amount_of_points):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_img = img

    # Geodesic distance edge indicator
    seg = perform_segmentation(
        image=img,
        initial_contours_coordinates=initial_countour_coordinates,
        iter_inner=iter_inner,
        iter_outer=iter_outer,
        lmbda=lmbda,
        alfa=alfa,
        epsilon=epsilon,
        sigma=sigma,
        potential_function=potential_function,
        edge_indicator=edge_indicator,
        amount_of_points=amount_of_points
    )


    # Euclidean distance edge indicator
    # seg = perform_segmentation(
    #     image=img,
    #     initial_contours_coordinates=[tuple([80, 265, 80, 250])],
    #     iter_inner=15,
    #     iter_outer=70,
    #     lmbda=2,
    #     alfa=4,
    #     epsilon=1.5,
    #     sigma=2.5,
    #     potential_function=PotentialFunction.DOUBLE_WELL,
    #     edge_indicator=EdgeIndicator.EUCLIDEAN_DISTANCE,
    #     amount_of_points=100
    # )

    # Scalar difference edge indicator
    # seg = perform_segmentation(
    #     image=img,
    #     initial_contours_coordinates=[tuple([80, 265, 80, 250])],
    #     iter_inner=15,
    #     iter_outer=50,
    #     lmbda=4,
    #     alfa=3,
    #     epsilon=1.5,
    #     sigma=1,
    #     potential_function=PotentialFunction.DOUBLE_WELL,
    #     edge_indicator=EdgeIndicator.SCALAR_DIFFERENCE,
    #     amount_of_points=100
    # )

    n = 0
    while True:
        try:
            phi = next(seg)
        except StopIteration:
            return_contour = convert_to_0_1(phi.tolist())
            return_contour_npy = np.array(return_contour)
            np.save('prediction.npy', return_contour_npy)
            #np.save('../try_out_segment_algorithms/array_data2.csv', return_contour_npy)
            return None
