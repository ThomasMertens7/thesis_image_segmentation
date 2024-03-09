import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import KFold

# Define your custom neural network architecture
# Should I fine tune a model or define a neural network from scratch here?
# Should then all images have the same size here?

import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted input size for the first fully connected layer
        self.fc1 = nn.Linear(64 * 80 * 80, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 8)

        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout probability

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 80 * 80)  # Flatten the output

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


def read_image(file_name):
    image = Image.open('../database/all_imgs/' + file_name + '.jpeg')
    return image


def update_df_with_images(df):
    image_data = []
    for file_name in df['image']:
        image = read_image(file_name)
        image_data.append(image)
    df['image'] = image_data
    return df


def dataframe_to_torch(df):
    transform = transforms.ToTensor()

    images = []
    labels = []

    max_width = 0
    max_height = 0

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Convert the PIL image to a tensor
        image_tensor = transform(row['image'])

        max_width = max(max_width, image_tensor.shape[2])
        max_height = max(max_width, image_tensor.shape[1])

        # Append the tensor to the list of images
        images.append(image_tensor)

        # Assuming 'edge_indicator', 'alpha', 'sigma', 'lambda', 'inner_iterations', 'outer_iterations' are your labels
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

        # Append the label tensor to the list of labels
        labels.append(label_tensor)

    padded_images = []
    for image_tensor in images:
        # Calculate padding lengths for each dimension
        pad_width = max_width - image_tensor.shape[2]
        pad_height = max_height - image_tensor.shape[1]

        # Ensure padding lengths are divisible by 2
        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left
        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top

        # Pad the image tensor
        padded_image = torch.nn.functional.pad(
            image_tensor,
            (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)
        )

        padded_images.append(padded_image)

    print(max_width)
    print(max_height)
    images_stack = torch.stack(padded_images)
    labels_stack = torch.stack(labels)
    # Stack the list of tensors to create a single tensor for images and labels
    return images_stack, labels_stack


excel_file_path = '../latest_data.xlsx'
df = pd.read_excel(excel_file_path)

condition = df['difficulty'] == 'easy'
filtered_rows_df = df[condition]

# Do I include the initial_contour? Or should I use the previously made model for determining the box?
# Do I include the precision?

df_without_index = filtered_rows_df.reset_index(drop=True)

df_with_images = update_df_with_images(df_without_index)

new_columns = {'edge_indicator': ["SCALAR_DIFFERENCE", "EUCLIDEAN_DISTANCE", "GEODESIC_DISTANCE"]}

df_with_images["SCALAR_DIFFERENCE"] = df_with_images["edge_indicator"]\
    .apply(lambda x: 1 if "EdgeIndicator.SCALAR_DIFFERENCE" in x else 0)
df_with_images["EUCLIDEAN_DISTANCE"] = df_with_images["edge_indicator"]\
    .apply(lambda x: 1 if "EdgeIndicator.EUCLIDEAN_DISTANCE" in x else 0)
df_with_images["GEODESIC_DISTANCE"] = df_with_images["edge_indicator"]\
    .apply(lambda x: 1 if "EdgeIndicator.GEODESIC_DISTANCE" in x else 0)

columns_to_select = ['image', 'SCALAR_DIFFERENCE', 'EUCLIDEAN_DISTANCE', 'GEODESIC_DISTANCE', 'alpha', 'sigma',
                     'lambda', 'inner_iterations', 'outer_iterations']
df_with_images = df_with_images[columns_to_select]

transform = transforms.ToTensor()

train_dataset = df_with_images[0:68]
train_data, train_labels = dataframe_to_torch(train_dataset)

batch_size = 2
num_epochs = 3

# Which loss function to use?
loss_function = nn.MSELoss()


# Train your model
# 1 classification parameter whilst the others are regression parameters, so make sure that the edge indicator is solved with bce
# Or should I round it?
# Or should I develop a separate model for it?
kf = KFold(n_splits=10, shuffle=True)
performance_metrics = []

for train_index, val_index in kf.split(train_dataset):
    model = CNNModel()
    optimizer = optim.SGD(model.parameters(), lr=0.002)

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
            print("training")

    with torch.no_grad():
        total_loss = 0
        total_samples = 0

        for batch in test_loader:
            images, labels = batch
            predicted_labels = model(images)
            print(predicted_labels)

            loss = loss_function(predicted_labels, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    average_loss = total_loss / total_samples
    print("Average loss: ", average_loss)

    performance_metrics.append(average_loss)

print(performance_metrics)











