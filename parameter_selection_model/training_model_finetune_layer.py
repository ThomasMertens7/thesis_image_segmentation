from torch.optim.lr_scheduler import StepLR
from transformers import AutoImageProcessor, ResNetModel
import torch
import torch.nn as nn
from preprocessing import preprocessing
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 8)

    def forward(self, x):
        # Formula for dimensions = (input_size - kernel_size + 2 x padding) / stride + 1
        # 512 x 7 x 7
        x = self.pool(torch.relu(self.conv1(x)))
        # 64 x 3 x 3
        x = self.pool(torch.relu(self.conv2(x)))
        # 32 x 1 x 1
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = ResNetModel.from_pretrained("facebook/vit-msn-small")

df = preprocessing()

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
        row['outer_iterations']
    ]))

train_data, train_labels = torch.stack(hidden_representations), torch.stack(labels)

batch_size = 1
num_epochs = 3

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train, X_val = train_data[0:61], train_data[61:69]
y_train, y_val = train_labels[0:61], train_labels[61:69]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)  # Reduce LR by a factor of 0.1 every 5 epochs

model = CNNModel()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(outputs)
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    scheduler.step()


