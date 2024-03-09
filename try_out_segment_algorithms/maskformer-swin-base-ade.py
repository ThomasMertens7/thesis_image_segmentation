from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import numpy as np
import cv2
from segmentinitialcontour.segment_initial_countour_train import get_initial_contour

name = "horse-40"
nb = 1


input_image = Image.open("../database/all_imgs/" + name + ".jpeg")

feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

inputs = feature_extractor(images=input_image, return_tensors="pt")

outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to feature_extractor for postprocessing
result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[input_image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
numpy_semantic_map = result["segmentation"].numpy()

all_elements = set()
for row in numpy_semantic_map:
    for element in row:
        all_elements.add(element)

print(nb)
class_mapping = dict()

index = 0
for elem in all_elements:
    class_mapping[elem] = index
    index += 1

mapped_numpy_semantic_map = np.vectorize(class_mapping.get)(numpy_semantic_map)



all_elements = set()
for row in mapped_numpy_semantic_map:
    for element in row:
        all_elements.add(element)

print(all_elements)

original_all_elements = all_elements.copy()
all_elements.remove(nb)

#print("all elements", all_elements)
#print("original all elements", original_all_elements)


class_mapping = dict()
for elem in original_all_elements:
    if elem in all_elements:
        class_mapping[elem] = 0
    else:
        class_mapping[elem] = 1
#print("class_mapping", class_mapping)
#np.savetxt('test.csv', mapped_numpy_semantic_map, delimiter=',')

final_mapped_numpy_semantic_map = np.vectorize(class_mapping.get)(mapped_numpy_semantic_map)
#print(np.sum(final_mapped_numpy_semantic_map))

#print(final_mapped_numpy_semantic_map)
np.save(name + '.npy', final_mapped_numpy_semantic_map)
#np.savetxt('array_data.csv', final_mapped_numpy_semantic_map, delimiter=',')


color_mapping = {
    0: (0, 0, 0),         # Class 0 is black
    1: (255, 0, 0),       # Class 1 is red
    2: (0, 255, 0),       # Class 2 is green
    3: (0, 0, 255),       # Class 3 is blue
    4: (255, 255, 0),     # Class 4 is yellow
    5: (255, 255, 255),   # Class 5 is white
    6: (0, 255, 255),     # Class 6 is cyan
    7: (128, 0, 0),       # Class 7 is maroon
    8: (0, 128, 0),       # Class 8 is dark green
    9: (0, 0, 128),       # Class 9 is navy
    10: (128, 128, 0),    # Class 10 is olive
    11: (128, 0, 128),    # Class 11 is purple
    12: (0, 128, 128),    # Class 12 is teal
    13: (192, 192, 192),  # Class 13 is silver
    14: (128, 128, 128),  # Class 14 is gray
    15: (255, 165, 0),    # Class 15 is orange
    16: (255, 192, 203),  # Class 16 is pink
    17: (210, 180, 140),  # Class 17 is tan
    18: (0, 255, 127),    # Class 18 is spring green
    19: (218, 112, 214)   # Class 19 is orchid
}

image = np.zeros((numpy_semantic_map.shape[0], numpy_semantic_map.shape[1], 3), dtype=np.uint8)
for class_label, color in color_mapping.items():
    # Set pixels with the corresponding class label to the color
    image[np.where(mapped_numpy_semantic_map == class_label)] = color

pil_image = Image.fromarray(image)

# Composite the background and foreground images with transparency
blended_image = Image.blend(input_image, pil_image, 0.3)

# Show the composite image
blended_image.show()


original_image = cv2.imread("../database/all_imgs/" + name + ".jpeg")
y_min, y_max, x_min, x_max = get_initial_contour("../database/all_imgs/" + name + ".jpeg")[0]

annotated_image = original_image.copy()
cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


