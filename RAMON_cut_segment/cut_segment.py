from PIL import Image
from math import floor
import os


def cut_image(phi, img_path):
    height = len(phi)
    width = len(phi[0])

    img_data = Image.open(img_path).convert("RGBA").getdata()

    new_img_data = []
    for index in range(width * height):
        if phi[floor(index/width)][index%width] > 0:
            new_img_data.append((img_data[index][0], img_data[index][1], img_data[index][2], 0))
        else:
            new_img_data.append(img_data[index])

    new_img = Image.new('RGBA', (width, height))
    new_img.putdata(new_img_data)
    new_path = os.path.splitext(img_path)[0] + "_segmented.png"
    new_img.save(new_path)

    return new_path


