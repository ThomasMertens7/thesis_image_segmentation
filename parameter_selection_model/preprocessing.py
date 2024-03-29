from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


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


def preprocessing():
    excel_file_path = '../pipeline/latest_data.xlsx'
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
    return df_with_images[columns_to_select]

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
    return df_with_images[columns_to_select]