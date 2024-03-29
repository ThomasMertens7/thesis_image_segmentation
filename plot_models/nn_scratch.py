import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel is a matrix of 9 weights, stride is the step size for every iteration,
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 40 * 40, 512)
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
        x = x.view(-1, 128 * 40 * 40)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

model = CNNModel()

dummy_input = torch.randn(3, 640, 640)

# Set up TensorBoard writer and add the model graph
writer = SummaryWriter('runs/scratch_visualizatzion')
writer.add_graph(model, dummy_input)
writer.close()