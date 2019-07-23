from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms
from helpers import visualize_dataset, fit, evaluate, get_data, load_model
from models import Classifier, FeatureModel, BagNet
import sys
import argparse
import unittest

class Tests(unittest.TestCase):
    def atestt_shape(self):
        bagnet = BagNet(10)
        img = torch.randn((3, 200, 200))
        patches = bagnet.img_to_patch(img)
        self.assertTrue(patches.shape == (20, 20, 3, 10, 10))
    
    def test_square_grid(self):
        img = cv2.imread('square_grid.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1)
        bagnet = BagNet(84)
        patch = bagnet.img_to_patch(img)

        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                img = patch[i][j].detach().permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.show()
        
                


if __name__ == '__main__':
    unittest.main()