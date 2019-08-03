from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms


class SimpleModel(nn.Module):
    def get_conv_layer(self, param):
        return nn.Sequential(*[nn.Conv2d(*param), nn.ReLU(), nn.BatchNorm2d(param[1])])
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.feature_extractor = nn.Sequential(*[
            self.get_conv_layer((1, 64, 3, 1, 1)), 
            nn.MaxPool2d(2), 
            self.get_conv_layer((64, 64, 3, 1, 1)), 
            nn.MaxPool2d(2),
            self.get_conv_layer((64, 1, 3, 1, 1))    
        ])
        self.fc1 = nn.Linear(49, 10)
    
    def forward(self, x):
        x1 = self.feature_extractor(x)
        return self.fc1(x1.view(-1, 49))

