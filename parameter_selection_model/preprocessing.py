from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import statistics
import numpy as np



def get_groups(df):
    all_tensors = []
    all_indexes = []

    index = -1
    for i, row in df.iterrows():
        if row['image'] not in all_tensors:
            index += 1
            all_tensors.append(row['image'])
            all_indexes.append(index)

        elif row['image'] in all_tensors:
            new_index = all_tensors.index(row['image'])
            all_indexes.append(new_index)

    return np.array(all_indexes)

def get_mean_and_var(df):
    alpha_mean = statistics.mean(list(df["alpha"]))
    alpha_st_dev = statistics.stdev(list(df["alpha"]))
    lmbda_mean = statistics.mean(list(df["lambda"]))
    lmbda_st_dev = statistics.stdev(list(df["lambda"]))
    sigma_mean = statistics.mean(list(df["sigma"]))
    sigma_st_dev = statistics.stdev(list(df["sigma"]))
    inner_iterations_mean = statistics.mean(list(df['inner_iterations']))
    inner_iterations_st_dev = statistics.stdev(list(df['inner_iterations']))
    outer_iterations_mean = statistics.mean(list(df["outer_iterations"]))
    outer_iterations_st_dev = statistics.stdev(list(df["outer_iterations"]))

    return [alpha_mean, alpha_st_dev, lmbda_mean, lmbda_st_dev, sigma_mean, sigma_st_dev, inner_iterations_mean,\
        inner_iterations_st_dev, outer_iterations_mean, outer_iterations_st_dev]


def normalize_list(lst, stats):
    if any(isinstance(item, list) for item in lst):
        for sublst in lst:
            sublst[3] = (sublst[3] - stats[0]) / stats[1]
            sublst[4] = (sublst[4] - stats[2]) / stats[3]
            sublst[5] = (sublst[5] - stats[4]) / stats[5]
            sublst[6] = (sublst[6] - stats[6]) / stats[7]
            sublst[7] = (sublst[7] - stats[8]) / stats[9]

    else:
        lst[3] = (lst[3] - stats[0]) / stats[1]
        lst[4] = (lst[4] - stats[2]) / stats[3]
        lst[5] = (lst[5] - stats[4]) / stats[5]
        lst[6] = (lst[6] - stats[6]) / stats[7]
        lst[7] = (lst[7] - stats[8]) / stats[9]

    return lst



def read_image(file_name):
    image = Image.open('../database/all_imgs/' + file_name + '.jpeg')
    return image

def update_df_with_images(df):
    image_data = []
    for file_name in df['image']:
        image = read_image(file_name)
        image_data.append(image)
    df['image'] = image_data
    return df


def preprocessing_filter(filter_name):
    excel_file_path = '../pipeline/latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['difficulty'] == 'easy'
    df = df[condition]
    filtered_rows_df = df[df['image'] != filter_name]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df_without_index = filtered_rows_df.reset_index(drop=True)

    df_with_images = update_df_with_images(df_without_index)

    new_columns = {'edge_indicator': ["SCALAR_DIFFERENCE", "EUCLIDEAN_DISTANCE", "GEODESIC_DISTANCE"]}

    df_with_images["SCALAR_DIFFERENCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df_with_images["EUCLIDEAN_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df_with_images["GEODESIC_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    """
    scaler = MinMaxScaler()
    scaler.fit(df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']] = scaler.transform(
        df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    """

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations']
    return df_with_images[columns_to_select]


def preprocessing_newer():
    excel_file_path = 'latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['predicted_diff'] == 'Easy'
    filtered_rows_df = df[condition]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df_without_index = filtered_rows_df.reset_index(drop=True)

    df_with_images = update_df_with_images(df_without_index)

    df_with_images["SCALAR_DIFFERENCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df_with_images["EUCLIDEAN_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df_with_images["GEODESIC_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    scaler = MinMaxScaler()
    scaler.fit(df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])
    df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']] = scaler.transform(
        df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points']
    df_with_images = df_with_images[columns_to_select]

    """
    mean_values = df_with_images[columns_to_scale].mean()
    std_values = df_with_images[columns_to_scale].std()

    print(mean_values)
    print(std_values)

    df_with_images[columns_to_scale] = (df_with_images[columns_to_scale] - mean_values) / std_values
    print(df_with_images[columns_to_scale])
    """

    return df_with_images


def preprocessing_newer_no_scaling():
    excel_file_path = 'latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['predicted_diff'] == 'Easy'
    filtered_rows_df = df[condition]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df_without_index = filtered_rows_df.reset_index(drop=True)

    df_with_images = update_df_with_images(df_without_index)

    df_with_images["SCALAR_DIFFERENCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df_with_images["EUCLIDEAN_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df_with_images["GEODESIC_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points']
    df_with_images = df_with_images[columns_to_select]

    """
    mean_values = df_with_images[columns_to_scale].mean()
    std_values = df_with_images[columns_to_scale].std()

    print(mean_values)
    print(std_values)

    df_with_images[columns_to_scale] = (df_with_images[columns_to_scale] - mean_values) / std_values
    print(df_with_images[columns_to_scale])
    """

    return df_with_images


def preprocessing():
    excel_file_path = 'old_newer_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['difficulty'] == 'easy'
    filtered_rows_df = df[condition]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df_without_index = filtered_rows_df.reset_index(drop=True)

    df_with_images = update_df_with_images(df_without_index)

    new_columns = {'edge_indicator': ["SCALAR_DIFFERENCE", "EUCLIDEAN_DISTANCE", "GEODESIC_DISTANCE"]}

    df_with_images["SCALAR_DIFFERENCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df_with_images["EUCLIDEAN_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df_with_images["GEODESIC_DISTANCE"] = df_with_images["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    """
    scaler = MinMaxScaler()
    scaler.fit(df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']] = scaler.transform(
        df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    """

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations']
    df_with_images = df_with_images[columns_to_select]

    columns_to_scale = ['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']

    """
    mean_values = df_with_images[columns_to_scale].mean()
    std_values = df_with_images[columns_to_scale].std()

    print(mean_values)
    print(std_values)

    df_with_images[columns_to_scale] = (df_with_images[columns_to_scale] - mean_values) / std_values
    print(df_with_images[columns_to_scale])
    """

    shuffled_index = np.random.permutation(df.index)
    return df.reindex(shuffled_index)


def preprocessing_features():
    excel_file_path = '../pipeline/latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['difficulty'] == 'easy'
    filtered_rows_df = df[condition]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df = filtered_rows_df.reset_index(drop=True)


    df["SCALAR_DIFFERENCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df["EUCLIDEAN_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df["GEODESIC_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    label_encoder = LabelEncoder()

    values = set()
    for val in df['direction']:
        values.add(val)



    df['animal'] = label_encoder.fit_transform(df['animal'])
    df['pose'] = label_encoder.fit_transform(df['pose'])
    df['direction'] = label_encoder.fit_transform(df['direction'])
    df['coverage'] = label_encoder.fit_transform(df['coverage'])


    """
    scaler = MinMaxScaler()
    scaler.fit(df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']] = scaler.transform(
        df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    """

    column_features = ['animal', 'pose', 'direction', 'coverage', 'brightness', 'color_variety']
    column_labels = ['SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations']
    return df[column_features], df[column_labels]

def preprocessing_with_animal():
    excel_file_path = '../pipeline/latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['difficulty'] == 'easy'
    filtered_rows_df = df[condition]

    # Do I include the initial_contour? Or should I use the previously made model for determining the box?
    # Do I include the precision?

    df_without_index = filtered_rows_df.reset_index(drop=True)

    df_with_images = update_df_with_images(df_without_index)

    """
    scaler = MinMaxScaler()
    scaler.fit(df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']] = scaler.transform(
        df_with_images[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']])
    """

    columns_to_select = ['image', 'edge_indicator', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'animal']
    df_with_images = df_with_images[columns_to_select]

    columns_to_scale = ['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations']

    """
    mean_values = df_with_images[columns_to_scale].mean()
    std_values = df_with_images[columns_to_scale].std()

    print(mean_values)
    print(std_values)

    df_with_images[columns_to_scale] = (df_with_images[columns_to_scale] - mean_values) / std_values
    print(df_with_images[columns_to_scale])
    """

    return df_with_images
