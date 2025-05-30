import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureCNN(nn.Module):
    def __init__(self):
        super(SeizureCNN, self).__init__()
        # 输入维度应为特征矩阵大小：例如 [batch_size, 1, 18, 18]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        
        # 展平后特征维度：16个通道，feature map 大小为 4x4 （取决于输入维度）
        self.fc1 = nn.Linear(16 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 2)  # 二分类：pre-ictal 和 inter-ictal

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
