from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import numpy as np
import gc
import torch


def get_ground_truth_segmentation(name):

    input_image = Image.open("../database/all_imgs_new/" + name + ".jpeg")

    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

    inputs = feature_extractor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    del model
    gc.collect()

    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[input_image.size[::-1]])[0]
    numpy_semantic_map = result["segmentation"].numpy()

    color_mapping = {
        0: (0, 0, 0),  # Class 0 is black
        1: (255, 0, 0),  # Class 1 is red
        2: (0, 255, 0),  # Class 2 is green
        3: (0, 0, 255),  # Class 3 is blue
        4: (255, 255, 0),  # Class 4 is yellow
        5: (128, 0, 128),  # Class 5 is purple
        6: (0, 255, 255),  # Class 6 is cyan
        7: (128, 0, 0),  # Class 7 is maroon
        8: (0, 128, 0),  # Class 8 is dark green
        9: (0, 0, 128),  # Class 9 is navy
        10: (128, 128, 0),  # Class 10 is olive
        11: (255, 255, 255),  # Class 11 is white
        12: (0, 128, 128),  # Class 12 is teal
        13: (192, 192, 192),  # Class 13 is silver
        14: (128, 128, 128),  # Class 14 is gray
        15: (255, 165, 0),  # Class 15 is orange
        16: (255, 192, 203),  # Class 16 is pink
        17: (210, 180, 140),  # Class 17 is tan
        18: (0, 255, 127),  # Class 18 is spring green
        19: (218, 112, 214)  # Class 19 is orchid
    }

    reverse_color_mapping = {
        "black": 0,  # Black is class 0
        "red": 1,  # Red is class 1
        "green": 2,  # Green is class 2
        "blue": 3,  # Blue is class 3
        "yellow": 4,  # Yellow is class 4
        "purple": 5,  # Purple is class 5 (originally mislabeled as white)
        "cyan": 6,  # Cyan is class 6
        "maroon": 7,  # Maroon is class 7
        "dark green": 8,  # Dark green is class 8
        "navy": 9,  # Navy is class 9
        "olive": 10,  # Olive is class 10
        "white": 11,  # Purple is class 11 (repeated, as both class 5 and 11 were originally purple)
        "teal": 12,  # Teal is class 12
        "silver": 13,  # Silver is class 13
        "gray": 14,  # Gray is class 14
        "orange": 15,  # Orange is class 15
        "pink": 16,  # Pink is class 16
        "tan": 17,  # Tan is class 17
        "spring green": 18,  # Spring green is class 18
        "orchid": 19  # Orchid is class 19
    }

    image = np.zeros((numpy_semantic_map.shape[0], numpy_semantic_map.shape[1], 3), dtype=np.uint8)
    for class_label, color in color_mapping.items():
        # Set pixels with the corresponding class label to the color
        image[np.where(numpy_semantic_map == class_label)] = color

    pil_image = Image.fromarray(image)

    # Composite the background and foreground images with transparency
    blended_image = Image.blend(input_image, pil_image, 0.3)
    blended_image.show()

    col = input("What is the target color? ")

    nb = reverse_color_mapping[col]

    all_elements = set()
    class_mapping = dict()

    for row in numpy_semantic_map:
        for element in row:
            all_elements.add(element)

    original_all_elements = all_elements.copy()
    all_elements.remove(nb)

    for elem in original_all_elements:
        if elem in all_elements:
            class_mapping[elem] = 0
        else:
            class_mapping[elem] = 1

    final_mapped_numpy_semantic = np.vectorize(class_mapping.get)(numpy_semantic_map)

    np.save('ground_truth.npy', final_mapped_numpy_semantic)