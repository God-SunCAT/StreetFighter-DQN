import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class LearningNet(nn.Module):
    def __init__(self, input_channels=4, action_size=18):
        super().__init__()
        
        # 4*84*84
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 4*84*84 -> 64*7*7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 对权重使用 Xavier Uniform
                init.xavier_uniform_(m.weight)
                # 如果有偏置，初始化为 0
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (batch, input_channels, 84, 84)
        # -> (batch, action)
        x = x / 255

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        
        return x