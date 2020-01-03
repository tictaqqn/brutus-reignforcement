from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = getLogger(__name__)


class BrutusModelPytorch(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 7, 5)

    def forward(self, x):
        pass
