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


class SeizureCNN_TH(nn.Module):
    def __init__(self):
        super(SeizureCNN_TH, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 输出: [B, 8, 2, 18]
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 输出: [B, 16, 2, 9]
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 不再池化，保留空间信息
        )

        # self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 2 * 9, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SeizureResNet(nn.Module):
    def __init__(self):
        super(SeizureResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.resblock1 = ResidualBlock(16, 16)
        self.resblock2 = ResidualBlock(16, 32, downsample=True)
        self.resblock3 = ResidualBlock(32, 32)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)
