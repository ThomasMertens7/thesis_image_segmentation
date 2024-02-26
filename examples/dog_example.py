import json

import numpy as np
import cv2
from geomeansegmentation.image_segmentation.drlse_segmentation import perform_segmentation, PotentialFunction, EdgeIndicator, construct_g
from figure.show_figure import show_all, show_lsf, show_contour
from segmentrecognition.image_recognition_recognize import recognize_animal
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour
from convert_contour import convert_to_0_1
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
            n = n + 1
            phi = next(seg)
            if n == 1:
                show_lsf(phi)
                show_contour(phi, original_img)
            else:
                show_all(phi, original_img)
        except StopIteration:
            return_contour = convert_to_0_1(phi.tolist())
            return_contour_npy = np.array(return_contour)
            np.save('../try_out_segment_algorithms/array_data2.npy', return_contour_npy)
            #np.savetxt('../try_out_segment_algorithms/array_data2.csv', return_contour_npy, delimiter=',')

            with open('./contours.json', 'w') as f:
                json.dump(return_contour, f)
            recognize_animal(img_path)
            while True:
                a = 0


if __name__ == "__main__":
    execute("../database/all_imgs/dog-1.jpeg", [[28, 206, 144, 256]], 15, 30, 0.5, 2.0, 1.5, 3, PotentialFunction.DOUBLE_WELL, EdgeIndicator.SCALAR_DIFFERENCE, 100)
