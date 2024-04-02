from perceptual_difference_unit import perceptual_difference_unit
from geomeansegmentation.image_segmentation.drlse_segmentation import PotentialFunction, EdgeIndicator
from examples.execute_example import execute
from get_optimal_parameters import get_optimal_parameters
from get_ground_truth_segmentation import get_ground_truth_segmentation
import pandas as pd
from ast import literal_eval
import gc


def get_precision():
    df = pd.read_excel('latest_data.xlsx')

    row = df.iloc[200]

    image_path = '../database/all_imgs/' + row['image'] + '.jpeg'

    execute(image_path, literal_eval(row['initial_contour']), row['inner_iterations'], row['outer_iterations'],
            row['lambda'], row['alpha'], 1.5, row['sigma'], PotentialFunction.DOUBLE_WELL,
            EdgeIndicator.from_string(row['edge_indicator']), row['num_points'])

    precision = perceptual_difference_unit('/Users/thomasmertens/Desktop/thesis/git_code'
                                               '/try_out_segment_algorithms/' +
                                               row['image'] + '.npy','prediction.npy')
    print(precision)
    print(image_path)
    gc.collect()


get_precision()
