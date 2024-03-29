from examples.execute_example import execute
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction, EdgeIndicator
from hamming_distance import hamming_distance
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour


def parameter_search(file_name):
    path = "../database/all_imgs/" + file_name + ".jpeg"

    bounding_box = get_initial_contour(path)

    old_alpha = None
    old_sigma = None
    old_outer = None
    old_inner = None
    old_lmbda = None

    max_prec = 0
    best_set = []

    for edge_indicator in [EdgeIndicator.GEODESIC_DISTANCE, EdgeIndicator.EUCLIDEAN_DISTANCE, EdgeIndicator.SCALAR_DIFFERENCE]:
        for alpha in [-0.75 + i * 0.25 for i in range(31)]:
            for sigma in [-0.75 + i * 0.25 for i in range(31)]:
                for outer_iter in [2 + i * 4 for i in range(15)]:
                    for inner_iter in [1 + i * 2 for i in range(15)]:
                        for lmbda in [- 0.75 + i * 0.25 for i in range(31)]:
                            if edge_indicator == EdgeIndicator.GEODESIC_DISTANCE:
                                for amount_of_points in [50, 62, 75, 87, 100, 112, 125, 137, 150]:
                                    print(path, bounding_box, inner_iter, outer_iter, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                                        edge_indicator, amount_of_points)
                                    execute(path, bounding_box, inner_iter, outer_iter, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                                        edge_indicator, amount_of_points)
                                    prec = hamming_distance(file_name + ".npy", "segmented.npy")
                                    if prec > max_prec:
                                        max_prec = prec
                                        best_set = [edge_indicator, alpha, sigma, outer_iter, inner_iter, lmbda, amount_of_points]

                            else:
                                print(path, bounding_box, inner_iter, outer_iter, lmbda, alpha, 1.5, sigma,
                                        PotentialFunction.DOUBLE_WELL,
                                        edge_indicator, 100)
                                execute(path, bounding_box, inner_iter, outer_iter, lmbda, alpha, 1.5, sigma,
                                        PotentialFunction.DOUBLE_WELL,
                                        edge_indicator, 100)
                                prec = hamming_distance(file_name + ".npy", "segmented.npy")
                                if prec > max_prec:
                                    max_prec = prec
                                    best_set = [edge_indicator, alpha, sigma, outer_iter, inner_iter, lmbda,
                                                100]
    return best_set, max_prec


parameter_search("dog-1")

