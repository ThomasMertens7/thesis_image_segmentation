from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from PIL import Image

def map_index_to_label(nb):
    if nb == 0:
      return "dog"
    elif nb == 1:
      return "horse"
    elif nb == 2:
      return "cow"
    elif nb == 3:
      return "butterfly"
    elif nb == 4:
      return "sheep"

def recognize_animal(img_path):
    model = AutoModelForImageClassification.from_pretrained("/Users/thomasmertens/Desktop/thesis/git_code/segmentrecognition/saved_model")
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    image = Image.open(img_path)

    inputs = feature_extractor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1)

    if predictions == "cow":
        pass
        # cow_example.execute()

    print("Predicted animal:", map_index_to_label(predictions.item()))