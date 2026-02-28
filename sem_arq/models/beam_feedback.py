import torch
import torch.nn as nn
import torch.nn.functional as F
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(128,256,3,stride=2,padding=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pooling(x)
        x = torch.flatten(1)
        return x
class Beam_Feedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

        ####没有加sigmoid，因为nn.BCEWithLogitsLoss()内置了sigmoid
        ######beam_feedback决策头：输出0/1表示Narrow/Widen
        self.beam_feedback = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, x):
        beam_logit = self.beam_feedback(x)
        return beam_logit
