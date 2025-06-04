import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureMLP(nn.Module):
    def __init__(self):
        super(SeizureMLP, self).__init__()
        self.flatten = nn.Flatten()  # 将输入 [B, 1, 2, 36] 展平为 [B, 72]

        self.fc1 = nn.Linear(72, 64)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.flatten(x)         # [B, 72]
        x = F.relu(self.fc1(x))  # [B, 64]
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.bn1(self.fc1(x)))  # [B, 128]
        # x = F.relu(self.bn2(self.fc2(x)))  # [B, 64]
        x = self.fc3(x)             # [B, 2]
        return F.softmax(x, dim=1)
