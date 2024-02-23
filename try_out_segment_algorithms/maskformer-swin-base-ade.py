from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

image = Image.open("../img/dog.jpeg")
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
inputs = feature_extractor(images=image, return_tensors="pt")

model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to feature_extractor for postprocessing
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# Convert the tensor to a NumPy array
numpy_semantic_map = predicted_semantic_map.numpy()

# Assuming you have an image corresponding to the semantic map
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Resize the semantic map to match the image size
numpy_semantic_map_resized = np.array(Image.fromarray(numpy_semantic_map[0]).resize(image.size, Image.NEAREST))

# Now you can access pixel values
pixel_values = numpy_semantic_map_resized[y, x]
print(pixel_values)
print(predicted_semantic_map)
pixel_decoder = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[1]
print(pixel_decoder)