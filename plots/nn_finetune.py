import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch

class FINETUNEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(75648, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        # Formula for dimensions = (input_size - kernel_size + 2 x padding) / stride + 1
        # 512 x 7 x 7
        x = x.view(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = FINETUNEModel()

dummy_input = torch.randn(1, 75648)

# Set up TensorBoard writer and add the model graph
writer = SummaryWriter('runs/model_visualization')
writer.add_graph(model, dummy_input)
writer.close()
