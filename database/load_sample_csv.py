import datasets
from PIL import Image

feature = datasets.Image(decode=False)


def transformToImg(example):
    print(type(Image.open(example['img'])))
    example['img'] = feature.encode_example(Image.open(example['img']))
    return example


local_csv_dataset = datasets.load_dataset("csv", data_files="easy.csv")

img_dataset = local_csv_dataset.cast_column('img', datasets.Image())

print(img_dataset)
print(img_dataset['train'][0]['img'])
