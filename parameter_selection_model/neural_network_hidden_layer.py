from torch.optim.lr_scheduler import StepLR
from transformers import AutoImageProcessor, ResNetModel, ViTMSNModel
import torch
import torch.nn as nn
from preprocessing import preprocessing, get_mean_and_var, normalize_list, get_groups, preprocessing_newer_no_scaling
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time
import torch.nn.functional as F
import pandas as pd

t1 = time.time()


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(75648, 256)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        # Formula for dimensions = (input_size - kernel_size + 2 x padding) / stride + 1
        # 512 x 7 x 7
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing_newer_no_scaling()

hidden_representations = []
labels = []

for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    hidden_representations.append(last_hidden_states[0])

    labels.append(torch.tensor([
        row['SCALAR_DIFFERENCE'],
        row['EUCLIDEAN_DISTANCE'],
        row['GEODESIC_DISTANCE'],
        row['alpha'],
        row['sigma'],
        row['lambda'],
        row['inner_iterations'],
        row['outer_iterations'],
        row['num_points']
    ]))

train_data, train_labels = torch.stack(hidden_representations), torch.stack(labels)

batch_size = 1
num_epochs = 20

criterion = nn.MSELoss()

kf = GroupKFold(n_splits=10)

groups = get_groups(df)

total_mse = []
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(train_data, groups=groups):
    model = CNNModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.75)

    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

    with torch.no_grad():
        total_loss = 0
        total_samples = 0

        for batch in test_loader:
            images, labels = batch
            predicted_labels = model(images)

            max_index = np.argmax(predicted_labels[indices_to_transform])
            predicted_labels[indices_to_transform] = 0
            predicted_labels[indices_to_transform[max_index]] = 1

            mse = mean_absolute_error(predicted_labels, labels[0])
            print(labels)
            print(predicted_labels)
            print(mse)

            total_mse.append(mse)

print(sum(total_mse)/len(total_mse))
print(np.var(total_mse))

t2 = time.time()

print(t2 - t1)



