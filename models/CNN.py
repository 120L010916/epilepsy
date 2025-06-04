import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureCNN(nn.Module):
    def __init__(self):
        super(SeizureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  # Input: [B, 1, 2, 36]
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))  # 更合理地只在频段方向池化
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 2 * 9, 84)  # 根据池化后实际尺寸来修改
        self.fc2 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))  # shape: [B, 6, 2, 18]
        x = self.dropout1(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))  # shape: [B, 16, 2, 9]
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # 展平 → B × (16*2*9)
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class CNN_Simple(nn.Module):
    def __init__(self):
        super(CNN_Simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)       # [B, 1, 2, 36] -> [B, 8, 2, 36]
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))                # -> [B, 8, 2, 18]

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)      # -> [B, 16, 2, 18]
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))                # -> [B, 16, 2, 9]

        self.fc1 = nn.Linear(16 * 2 * 9, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class SeizureCNN1D(nn.Module):
    def __init__(self):
        super(SeizureCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # self.fc1 = nn.Linear(16 * 18, 64)  # 36 → 18 → 9
        self.fc2 = nn.Linear(16 * 18, 2)

    def forward(self, x):
        # x: [B, 1, 2, 36] → reshape 为 [B, 2, 36]
        x = x.squeeze(1)  # [B, 2, 36]
        x = F.relu(self.conv1(x))  # → [B, 16, 36]
        x = self.pool1(x)          # → [B, 16, 18]
        # x = F.relu(self.conv2(x))  # → [B, 32, 18]
        # x = self.pool2(x)          # → [B, 32, 9]
        x = x.view(x.size(0), -1)  # Flatten
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
