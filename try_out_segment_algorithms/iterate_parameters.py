from examples.execute_example import execute
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction, EdgeIndicator
from hamming_distance import hamming_distance
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour


def parameter_search(file_name, level=0):
    highest_prec = 0
    path = "../database/all_imgs/" + file_name + ".jpeg"

    bounding_box = get_initial_contour(path)
    diffs = [None, None, None]

    edge_indicator = None
    alpha = None
    sigma = None
    lmbda = None

    while True:
        # Determine optimal edge indicator
        if level == 0:
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.GEODESIC_DISTANCE, 100)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.SCALAR_DIFFERENCE, 100)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.EUCLIDEAN_DISTANCE, 100)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                edge_indicator = EdgeIndicator.GEODESIC_DISTANCE
                edge_indicator_name = "EdgeIndicator.GEODESIC_DISTANCE"
            elif prec2 >= prec1 and prec2 >= prec3:
                edge_indicator = EdgeIndicator.SCALAR_DIFFERENCE
                edge_indicator_name = "EdgeIndicator.SCALAR_DIFFERENCE"
            else:
                edge_indicator = EdgeIndicator.EUCLIDEAN_DISTANCE
                edge_indicator_name = "EdgeIndicator.EUCLIDEAN_DISTANCE"
            level = 1
            print(edge_indicator_name)

        # Determine optimal value of alpha
        elif level == 1:
            if diffs[0] is None:
                alpha = 3
                diffs[0] = 2

            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha + diffs[0], 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha - diffs[0], 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                alpha = alpha
            elif prec2 >= prec1 and prec2 >= prec3:
                alpha = alpha + diffs[0]
            else:
                alpha = alpha - diffs[0]

            if diffs[0] <= 0.125:
                level = 2

            diffs[0] = diffs[0] / 2

        # Determine optimal value of sigma
        elif level == 2:
            if diffs[1] is None:
                sigma = 3
                diffs[1] = 2

            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma + diffs[1], PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma - diffs[1], PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                sigma = sigma
            elif prec2 >= prec1 and prec2 >= prec3:
                sigma = sigma + diffs[1]
            else:
                sigma = sigma - diffs[1]

            if diffs[1] <= 0.125:
                level = 3

            diffs[1] = diffs[1] / 2


        # Determine optimal value of lambda
        elif level == 3:
            if diffs[2] is None:
                lmbda = 3
                diffs[2] = 2

            execute(path, bounding_box, 15, 30, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, lmbda + diffs[2], alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, lmbda - diffs[2], alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, 100)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                lmbda = lmbda
            elif prec2 >= prec1 and prec2 >= prec3:
                lmbda = lmbda + diffs[2]
            else:
                lmbda = lmbda - diffs[2]

            if diffs[2] <= 0.125:
                level = 4

            diffs[2] = diffs[2] / 2

        else:
            with open('results.txt', 'a') as file:
                file.write(file_name + ", " + str(bounding_box) + ", " + edge_indicator_name + ", " + str(alpha) + ", "
                           + str(sigma) + ", " + str(lmbda) + ", " + str(highest_prec) + "\n")
            return path, (edge_indicator, alpha, sigma, lmbda), highest_prec



for index in range(50, 51):
    print(index)
    parameter_search("horse-" + str(index), 0)
