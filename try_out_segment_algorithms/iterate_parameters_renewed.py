from examples.execute_example import execute
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction, EdgeIndicator
from hamming_distance import hamming_distance
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour


def parameter_search(file_name, level=0):
    highest_prec = 0
    path = "../database/all_imgs/" + file_name + ".jpeg"

    bounding_box = get_initial_contour(path)
    diffs = [None, None, None, None, None, None]

    edge_indicator = None
    alpha = None
    sigma = None
    lmbda = None
    outer_iter = None
    inner_iter = None
    amount_of_points = 100

    while True:
        # Determine optimal edge indicator
        if level == 0:
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.GEODESIC_DISTANCE, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.GEODESIC_DISTANCE, amount_of_points + 50)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.GEODESIC_DISTANCE, amount_of_points - 50)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.SCALAR_DIFFERENCE, amount_of_points)
            prec4 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    EdgeIndicator.EUCLIDEAN_DISTANCE, amount_of_points)
            prec5 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3 and prec1 >= prec4 and prec1 >= prec5:
                edge_indicator = EdgeIndicator.GEODESIC_DISTANCE
                edge_indicator_name = "EdgeIndicator.GEODESIC_DISTANCE"
            elif prec2 >= prec3 and prec2 >= prec4 and prec2 >= prec5:
                edge_indicator = EdgeIndicator.GEODESIC_DISTANCE
                edge_indicator_name = "EdgeIndicator.GEODESIC_DISTANCE"
                amount_of_points = int(amount_of_points + 50)
            elif prec3 >= prec4 and prec3 >= prec5:
                edge_indicator = EdgeIndicator.GEODESIC_DISTANCE
                edge_indicator_name = "EdgeIndicator.GEODESIC_DISTANCE"
                amount_of_points = int(amount_of_points - 50)
            elif prec4 >= prec5:
                edge_indicator = EdgeIndicator.SCALAR_DIFFERENCE
                edge_indicator_name = "EdgeIndicator.SCALAR_DIFFERENCE"
            else:
                edge_indicator = EdgeIndicator.EUCLIDEAN_DISTANCE
                edge_indicator_name = "EdgeIndicator.EUCLIDEAN_DISTANCE"

            level = 1
            print(edge_indicator_name)

        if level == 1:
            if edge_indicator == EdgeIndicator.GEODESIC_DISTANCE:

                if diffs[5] is None:
                    alpha = 3
                    diffs[5] = 25

                execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                        EdgeIndicator.GEODESIC_DISTANCE, int(amount_of_points))
                prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
                execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                        EdgeIndicator.GEODESIC_DISTANCE, int(amount_of_points + diffs[5]))
                prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
                execute(path, bounding_box, 15, 30, 3, 3, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                        EdgeIndicator.GEODESIC_DISTANCE, int(amount_of_points - diffs[5]))
                prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
                highest_prec = max(highest_prec, prec1, prec2, prec3)

                if prec1 >= prec2 and prec1 >= prec3:
                    amount_of_points = amount_of_points
                elif prec2 >= prec3:
                    amount_of_points = amount_of_points + diffs[5]
                else:
                    amount_of_points = amount_of_points - diffs[5]

                if diffs[5] <= 12.5:
                    level = 2

                diffs[5] = int(diffs[5] / 2)

        # Determine optimal value of alpha
        elif level == 2:
            if diffs[0] is None:
                alpha = 3
                diffs[0] = 2

            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha + diffs[0], 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha - diffs[0], 1.5, 3, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                alpha = alpha
            elif prec2 >= prec1 and prec2 >= prec3:
                alpha = alpha + diffs[0]
            else:
                alpha = alpha - diffs[0]

            if diffs[0] <= 0.25:
                level = 3

            diffs[0] = diffs[0] / 2

        # Determine optimal value of sigma
        elif level == 3:
            if diffs[1] is None:
                sigma = 3
                diffs[1] = 2

            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma + diffs[1], PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, 30, 3, alpha, 1.5, sigma - diffs[1], PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                sigma = sigma
            elif prec2 >= prec1 and prec2 >= prec3:
                sigma = sigma + diffs[1]
            else:
                sigma = sigma - diffs[1]

            if diffs[1] <= 0.25:
                level = 4

            diffs[1] = diffs[1] / 2

        # Determine optimal value of outer_iter
        elif level == 4:
            if diffs[2] is None:
                outer_iter = 30
                diffs[2] = 15

            execute(path, bounding_box, 15, outer_iter, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, outer_iter + diffs[2], 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15, outer_iter - diffs[2], 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                outer_iter = outer_iter
            elif prec2 >= prec1 and prec2 >= prec3:
                outer_iter = outer_iter + diffs[2]
            else:
                outer_iter = outer_iter - diffs[2]

            if diffs[2] <= 3.75:
                level = 5

            diffs[2] = int(diffs[2] / 2)

        # Determine optimal value of inner_iter
        elif level == 5:
            if diffs[3] is None:
                inner_iter = 15
                diffs[3] = 7.5

            execute(path, bounding_box, 15, outer_iter, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15 + diffs[3], outer_iter, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, 15 - diffs[3], outer_iter, 3, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                outer_iter = outer_iter
            elif prec2 >= prec1 and prec2 >= prec3:
                outer_iter = outer_iter + diffs[3]
            else:
                outer_iter = outer_iter - diffs[3]
            if diffs[3] <= 1.875:
                level = 6

            diffs[3] = int(diffs[3] / 2)

        # Determine optimal value of lambda
        elif level == 6:
            if diffs[4] is None:
                lmbda = 3
                diffs[4] = 2

            execute(path, bounding_box, inner_iter, outer_iter, lmbda, alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec1 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, inner_iter, outer_iter, lmbda + diffs[2], alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec2 = hamming_distance(file_name + ".npy", "segmented.npy")
            execute(path, bounding_box, inner_iter, outer_iter, lmbda - diffs[2], alpha, 1.5, sigma, PotentialFunction.DOUBLE_WELL,
                    edge_indicator, amount_of_points)
            prec3 = hamming_distance(file_name + ".npy", "segmented.npy")
            highest_prec = max(highest_prec, prec1, prec2, prec3)

            if prec1 >= prec2 and prec1 >= prec3:
                lmbda = lmbda
            elif prec2 >= prec1 and prec2 >= prec3:
                lmbda = lmbda + diffs[2]
            else:
                lmbda = lmbda - diffs[2]

            if diffs[2] <= 0.25:
                level = 7

            diffs[2] = diffs[2] / 2

        else:
            with open('results.txt', 'a') as file:
                file.write(file_name + ", " + str(bounding_box) + ", " + edge_indicator_name + ", " + str(outer_iter)
                           + ", " + str(inner_iter) + ", " + str(alpha) + ", "
                           + str(sigma) + ", " + str(lmbda) + ", " + str(amount_of_points) + ", " +
                           str(highest_prec) + "\n")

            return path, (edge_indicator, alpha, sigma, lmbda), highest_prec



for index in range(1, 51):
    print(index)
    parameter_search("dog-" + str(index), 0)
