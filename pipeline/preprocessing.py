import pandas as pd
from PIL import Image


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
def preprocessing():
    excel_file_path = 'latest_data.xlsx'
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

