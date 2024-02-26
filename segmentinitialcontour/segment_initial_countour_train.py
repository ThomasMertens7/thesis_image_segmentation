from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, YolosForObjectDetection, YolosImageProcessor
import torch
from datasets import load_dataset, load_from_disk, load_metric
import torch
import numpy as np
from PIL import Image
from math import floor, ceil


def convert_to_og_contour(contour):
    return [[floor(contour[1]), ceil(contour[3]), floor(contour[0]), ceil(contour[2])]]

def get_initial_contour(path):
    img = Image.open(path)

    model_name_or_path = 'hustvl/yolos-tiny'
    processor = YolosForObjectDetection.from_pretrained(model_name_or_path)


    model = YolosForObjectDetection.from_pretrained(model_name_or_path)
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes


    target_sizes = torch.tensor([img.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

    print(convert_to_og_contour(box))
    return convert_to_og_contour(box)



