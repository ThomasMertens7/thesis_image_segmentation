import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


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


def preprocessing_cv():
    excel_file_path = 'latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['predicted_diff'] == 'Easy'
    df = df[condition]
    df = df.reset_index(drop=True)

    df['name'] = df['image']
    df = update_df_with_images(df)

    df["SCALAR_DIFFERENCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df["EUCLIDEAN_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df["GEODESIC_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    scaler = MinMaxScaler()
    scaler.fit(df[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])
    df[
        ['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']] = scaler.transform(
        df[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                         'lambda', 'inner_iterations', 'outer_iterations', 'num_points']

    return df[columns_to_select]

def preprocessing_hard():
    excel_file_path = 'latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['predicted_diff'] == 'Hard'
    df = df[condition]
    df = df.reset_index(drop=True)

    df['name'] = df['image']
    df = update_df_with_images(df)

    columns_to_select = ['name', 'image']

    return df[columns_to_select]


def preprocessing():
    excel_file_path = 'latest_data.xlsx'
    df = pd.read_excel(excel_file_path)

    condition = df['predicted_diff'] == 'Easy'
    df = df[condition]
    df = df.reset_index(drop=True)

    df = update_df_with_images(df)

    df["SCALAR_DIFFERENCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
    df["EUCLIDEAN_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
    df["GEODESIC_DISTANCE"] = df["edge_indicator"] \
        .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

    scaler = MinMaxScaler()
    scaler.fit(df[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])
    df[
        ['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']] = scaler.transform(
        df[['alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']])

    columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha',
                         'sigma', 'lambda', 'inner_iterations', 'outer_iterations', 'num_points']

    return df[columns_to_select]

