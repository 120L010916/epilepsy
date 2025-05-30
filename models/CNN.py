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
