from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, ViTImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset, load_from_disk, load_metric
import torch
import numpy as np
from PIL import Image


def map_label_to_index(example):
    if example['labels'] == 'dog':
        example['labels'] = 0
    elif example['labels'] == 'horse':
        example['labels'] = 1
    elif example['labels'] == 'cow':
        example['labels'] = 2
    elif example['labels'] == 'butterfly':
        example['labels'] = 3
    elif example['labels'] == 'sheep':
        example['labels'] = 4
    return example


dataset = load_from_disk("/Users/thomasmertens/Desktop/thesis/git_code/segmentrecognition/train")
dataset = dataset.select(range(20000))
print(dataset)
print(type(dataset[1]["image"]))
dataset = dataset.remove_columns(['l14_embeddings', 'moco_vitb_imagenet_embeddings', 'moco_vitb_imagenet_embeddings_without_last_layer'])
dataset = dataset.rename_column('caption', 'labels')
desired_animals = ['dog', 'horse', 'cow', 'butterfly', 'sheep']
filtered_dataset = dataset.filter(lambda example: example['labels'] in desired_animals and example['image'].getbands() == ('R', 'G', 'B'))
filtered_dataset = filtered_dataset.map(map_label_to_index)
print(len(filtered_dataset))

shuffled_dataset = filtered_dataset.shuffle(seed=42)

model_name_or_path = 'google/vit-base-patch16-224-in21k'

processor = ViTImageProcessor.from_pretrained(model_name_or_path)


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


prepared_dataset = shuffled_dataset.with_transform(transform)
train_test_dataset = prepared_dataset.train_test_split(test_size=0.1)
training_dataset = train_test_dataset["train"]
test_valid_dataset = train_test_dataset["test"].train_test_split(test_size=0.5)
validation_dataset = test_valid_dataset["train"]
test_dataset = test_valid_dataset["test"]


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(desired_animals),
    id2label={str(i): c for i, c in enumerate(desired_animals)},
    label2id={c: str(i) for i, c in enumerate(desired_animals)}
)


training_args = TrainingArguments(
    output_dir="./vit",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=1,
    fp16=False,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=4e-4,
    save_total_limit=1,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=['tensorboard'],
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


