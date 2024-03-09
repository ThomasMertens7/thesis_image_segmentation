import cv2
import numpy as np
import pandas as pd

data = []

for index in range(1, 51):
    image = cv2.imread('all_imgs/horse-' + str(index) + '.jpeg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    average_intensity = 0
    num_pixels = gray_image.shape[0] * gray_image.shape[1]

    means, stds = cv2.meanStdDev(image_rgb)

    mean_r, mean_g, mean_b = means
    std_r, std_g, std_b = stds

    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            average_intensity += gray_image[i, j]


    print("Image " + str(index))
    average_intensity /= num_pixels
    print("Average intensity:", average_intensity)

    color_variance = np.mean([std_r, std_g, std_b])
    print("Color variance:", color_variance)

    data.append([average_intensity, color_variance])

#Colummns
columns = ['intensity', 'color variance']

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Write the DataFrame to an Excel file
df.to_excel('data.xlsx', index=False)