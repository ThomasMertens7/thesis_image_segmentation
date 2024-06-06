import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import torch
from torchvision import transforms
from sklearn.model_selection import KFold
from preprocessing import preprocessing, get_mean_and_var, normalize_list
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import time
import pandas as pd


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel is a matrix of 9 weights, stride is the step size for every iteration,
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(40 * 40 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 8)

        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout probability

    def forward(self, x):
        # Formula for dimensions = (input_size - kernel_size + 2 x padding) / stride + 1
        # 640 x 640 x 3
        x = F.relu(self.conv1(x))
        # 640 x 640 x 16
        x = self.pool(x)
        # 320 x 320 x 16
        x = F.relu(self.conv2(x))
        # 320 x 320 x 32
        x = self.pool(x)
        # 160 x 160 x 32
        x = F.relu(self.conv3(x))
        # 160 x 160 x 64
        x = self.pool(x)
        # 80 x 80 x 64
        x = F.relu(self.conv4(x))  # New convolutional layer
        # 80 x 80 x 128
        x = self.pool(x)
        # 40 x 40 x 128

        # Reshapes to long vector
        x = x.view(-1, 40 * 40 * 128)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x


def dataframe_to_torch(df):
    transform = transforms.ToTensor()

    images = []
    labels = []

    max_width = 0
    max_height = 0

    for index, row in df.iterrows():
        image_tensor = transform(row['image'])

        max_width = max(max_width, image_tensor.shape[2])
        max_height = max(max_width, image_tensor.shape[1])

        images.append(image_tensor)

        label_tensor = torch.tensor([
            row['SCALAR_DIFFERENCE'],
            row['EUCLIDEAN_DISTANCE'],
            row['GEODESIC_DISTANCE'],
            row['alpha'],
            row['sigma'],
            row['lambda'],
            row['inner_iterations'],
            row['outer_iterations']
        ])

        labels.append(label_tensor)

    padded_images = []
    for image_tensor in images:
        pad_width = max_width - image_tensor.shape[2]
        pad_height = max_height - image_tensor.shape[1]

        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left
        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top

        padded_image = torch.nn.functional.pad(
            image_tensor,
            (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)
        )

        padded_images.append(padded_image)

    print(max_width)
    print(max_height)
    images_stack = torch.stack(padded_images)
    labels_stack = torch.stack(labels)
    return images_stack, labels_stack

t1 = time.time()

df = preprocessing()

# Assuming 'df' is your DataFrame and 'columns_to_scale' is a list of columns you want to z-scale
columns_to_scale = ['column1', 'column2', 'column3']

# Calculate mean and standard deviation for each column to scale
mean_values = df[columns_to_scale].mean()
std_values = df[columns_to_scale].std()

# Z-scale the selected columns
df[columns_to_scale] = (df[columns_to_scale] - mean_values) / std_values

transform = transforms.ToTensor()

train_dataset = df[0:68]
train_data, train_labels = dataframe_to_torch(train_dataset)

batch_size = 1
num_epochs = 1

loss_function = nn.MSELoss()

kf = KFold(n_splits=10, shuffle=True)

total_mse = []
indices_to_transform = [0, 1, 2]

for train_index, val_index in kf.split(train_dataset):
    model = CNNModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.75)  # Reduce LR by a factor of 0.1 every 5 epochs

    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            images, labels = batch

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
    t2 = time.time()
    print(t2-t1)

    with torch.no_grad():
        total_loss = 0
        total_samples = 0

        for batch in test_loader:
            images, labels = batch
            predicted_labels = model(images)

            for row in predicted_labels:
                max_index = np.argmax(row[indices_to_transform])
                row[indices_to_transform] = 0
                row[indices_to_transform[max_index]] = 1

            stats = get_mean_and_var(pd.DataFrame(y_train.numpy(),
                                                  columns=["GEODESIC_DISTANCE", "EUCLIDEAN_DISTANCE",
                                                           "SCALAR_DIFFERENCE", "alpha", "sigma", "lambda",
                                                           "inner_iterations", "outer_iterations"]))

            mse = mean_squared_error(normalize_list(labels.numpy().tolist(), stats),
                                     normalize_list(predicted_labels.numpy().tolist(), stats))

            print(mse)
            total_mse.append(mse)

t2 = time.time()
print(sum(total_mse) / len(total_mse))
print(np.var(total_mse))
print(t2 - t1)













