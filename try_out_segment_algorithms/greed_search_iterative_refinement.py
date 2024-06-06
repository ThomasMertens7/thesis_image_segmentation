from RAMON_examples.execute_example import execute
from RAMON_geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction, EdgeIndicator
from hamming_distance import hamming_distance
from pipeline.perceptual_difference_unit import perceptual_difference_unit
from RAMON_segmentinitialcontour.segment_initial_countour_train import get_initial_contour


def optimize_key(path, box, edge_indicator, key, params):
    f_params = params.copy()
    evaluation_vals = []

    start_value = None
    delta = None
    search_length = None

    if key == "alpha" or key == "sigma" or key == "lambda":
        start_value = -0.75
        delta = 0.25
        search_length = 31
    elif key == "outer_iterations":
        start_value = 2
        delta = 4
        search_length = 15
    elif key == "inner_iterations":
        start_value = 1
        delta = 2
        search_length = 15
    elif key == "num_points":
        start_value = 50
        delta = 25
        search_length = 5

    for index in range(search_length):
        f_params[key] = start_value + delta * index
        execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"], f_params["alpha"],
                1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
        evaluation_vals.append(perceptual_difference_unit("dog-8.npy", "prediction.npy"))

    return start_value + evaluation_vals.index(max(evaluation_vals)) * delta


def update_key(path, box, edge_indicator, key, params):
    f_params = params.copy()

    start_value = None
    delta = None
    end_value = None

    if key == "alpha" or key == "sigma" or key == "lambda":
        start_value = -0.75
        delta = 0.25
        end_value = 6.75
    elif key == "outer_iterations":
        start_value = 2
        delta = 4
        end_value = 58
    elif key == "inner_iterations":
        start_value = 1
        delta = 2
        end_value = 29
    elif key == "num_points":
        start_value = 50
        delta = 25
        end_value = 150

    best_params = f_params[key]
    execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"], f_params["alpha"],
            1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
    best_result = perceptual_difference_unit("dog-8.npy", "prediction.npy")

    f_params[key] = f_params[key] + delta
    execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"], f_params["alpha"],
            1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
    new_result = perceptual_difference_unit("dog-8.npy", "prediction.npy")

    while new_result > best_result and end_value >= f_params[key] >= start_value:
        best_result = new_result
        best_params = f_params[key]

        f_params[key] = f_params[key] + delta
        execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"], f_params["alpha"],
                1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
        new_result = perceptual_difference_unit("dog-8.npy", "prediction.npy")
    else:
        f_params[key] = f_params[key] - 2*delta
        execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"], f_params["alpha"],
                1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
        new_result = perceptual_difference_unit("dog-8.npy", "prediction.npy")
        while new_result > best_result and end_value >= f_params[key] >= start_value:
            best_result = new_result
            best_params = f_params[key]

            f_params[key] = f_params[key] - delta
            execute(path, box, f_params["inner_iterations"], f_params["outer_iterations"], f_params["lambda"],
                    f_params["alpha"],
                    1.5, f_params["sigma"], PotentialFunction.DOUBLE_WELL, edge_indicator, f_params["num_points"])
            new_result = perceptual_difference_unit("dog-8.npy", "prediction.npy")

    return best_params


def parameter_search(file_name):
    path = "../database/all_imgs/" + file_name + ".jpeg"
    bounding_box = get_initial_contour(path)

    for edge_indicator in [EdgeIndicator.GEODESIC_DISTANCE, EdgeIndicator.EUCLIDEAN_DISTANCE,
                           EdgeIndicator.SCALAR_DIFFERENCE]:
        params = {"alpha": 3, "sigma": 3, "lambda": 3, "outer_iterations": 30,
                  "inner_iterations": 15, "num_points": 100}

        if edge_indicator == EdgeIndicator.GEODESIC_DISTANCE:
            order = ["alpha", "sigma", "lambda", "outer_iterations", "inner_iterations", "num_points"]
        else:
            order = ["alpha", "sigma", "lambda", "outer_iterations", "inner_iterations"]

        finished_params = []

        for key in order:
            print(key)
            print(1)
            print(params)
            params[key] = optimize_key(path, bounding_box, edge_indicator, key, params)
            print(2)
            print(params)
            for parameter in finished_params:
                print(parameter)
                params[parameter] = update_key(path, bounding_box, edge_indicator, parameter, params)
                print(3)
                print(params)
            finished_params.append(key)

        print(params)

parameter_search("dog-8")
