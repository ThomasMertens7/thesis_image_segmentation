"""
This Python package is written in the context of the Master's thesis of Robbe Ramon, KU Leuven.

Inspired from the following references:

    [1] Y. Rathi, A. Tannenbaum and O. Michailovich, "Segmenting Images on the Tensor Manifold," 2007 IEEE Conference on
    Computer Vision and Pattern Recognition, Minneapolis, MN, USA, 2007, pp. 1-8, doi: 10.1109/CVPR.2007.383010.

    [2] C. Li, C. Xu, C. Gui and M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image
    Segmentation," in IEEE Transactions on Image Processing, vol. 19, no. 12, pp. 3243-3254, Dec. 2010,
    doi: 10.1109/TIP.2010.2069690.

    [3] Bini, D.A., Iannazzo, B. A note on computing matrix geometric means. Advances in Computational Mathematics 35,
    175–192 (2011). https://doi.org/10.1007/s10444-010-9165-0

Author: Robbe Ramon
Released under MIT license
"""

import cv2
from geomeansegmentation.image_segmentation.drlse_segmentation import perform_segmentation, PotentialFunction, \
    EdgeIndicator
from figure.show_figure import show_all, show_lsf, show_contour


def main():
    img = cv2.imread("img/lizard.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_img = img

    # Geodesic distance edge indicator
    seg = perform_segmentation(
        image=img,
        initial_contours_coordinates=[tuple([95, 450, 80, 250])],
        iter_inner=15,
        iter_outer=67,
        lmbda=5,
        alfa=7,
        epsilon=1.5,
        sigma=2,
        potential_function=PotentialFunction.DOUBLE_WELL,
        edge_indicator=EdgeIndicator.GEODESIC_DISTANCE,
        amount_of_points=100
    )

    # Scalar difference edge indicator
    # seg = perform_segmentation(
    #     image=img,
    #     initial_contours_coordinates=[tuple([95, 450, 80, 250])],
    #     iter_inner=15,
    #     iter_outer=45,
    #     lmbda=7,
    #     alfa=3,
    #     epsilon=1.5,
    #     sigma=3,
    #     potential_function=PotentialFunction.DOUBLE_WELL,
    #     edge_indicator=EdgeIndicator.SCALAR_DIFFERENCE,
    #     amount_of_points=100
    # )

    # Euclidean distance edge indicator
    # seg = perform_segmentation(
    #     image=img,
    #     initial_contours_coordinates=[tuple([95, 450, 80, 250])],
    #     iter_inner=15,
    #     iter_outer=60,
    #     lmbda=5,
    #     alfa=7,
    #     epsilon=1.5,
    #     sigma=3,
    #     potential_function=PotentialFunction.DOUBLE_WELL,
    #     edge_indicator=EdgeIndicator.EUCLIDEAN_DISTANCE,
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
            break


if __name__ == "__main__":
    main()
