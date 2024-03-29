import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from transformers import AutoImageProcessor

from parameter_selection_model.preprocessing import preprocessing


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(102400, 128)

    def forward_one(self, x):
        # 3 x 224 x 224
        x = self.conv1(x)
        # 64 x 215 x 215
        x = self.relu(x)
        x = self.pool(x)
        # 64 x 107 x 107
        x = self.conv2(x)
        # 128 x 101 x 101
        x = self.relu(x)
        x = self.pool(x)
        # 128 x 50 x 50
        x = self.conv3(x)
        # 128 x 47 x 47
        x = self.relu(x)
        x = self.pool(x)
        # 128 x 23 x 23
        x = self.conv4(x)
        # 256 x 20 x 20
        x = self.relu(x)
        x = x.view(-1, 256 * 20 * 20)
        # (256 x 400)

        x = self.fc(x)
        x = self.sig(x)

        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = euclidean_distance
        return loss_contrastive



processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
df = preprocessing()

image_inputs = []


# Initialize network
model = SiameseNetwork()


for i, row in df.iterrows():
    inputs = processor(images=row['image'], return_tensors="pt")["pixel_values"][0]
    res1, res2 = model(inputs, inputs)
    image_inputs.append(inputs)


# Loss and optimizer
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
dataloader = None


for epoch in range(num_epochs):
    for img1, img2, label in dataloader:
        # Forward pass
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()