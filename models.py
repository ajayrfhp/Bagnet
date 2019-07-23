from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms


class FeatureModel(nn.Module):
    def get_conv_layer(self, param):
        return nn.Sequential(*[nn.Conv2d(*param), nn.BatchNorm2d(param[1]), nn.ReLU()])

    def __init__(self):
        '''
        Define model that maps (3, Q, Q) to (V)        
        '''
        super(FeatureModel, self).__init__()
        self.l1 = self.get_conv_layer((1, 64, 3))
        self.l2 = self.get_conv_layer((64, 64, 3))
        self.l3 = self.get_conv_layer((64, 128, 3))
        self.l = nn.Sequential(*[self.l1, self.l2, self.l3])
        self.fc1 = nn.Linear(128*22*22, 100)

        
    def forward(self, x):
        '''
        Args
            patch - (3, Q, Q)
        Returns
            representation - (V)
        Convolutional neural network that maps patch to a vector
        '''
        x1 = self.l(x)
        return self.fc1(x1.view(-1, 128*22*22))

class Classifier(nn.Module):
    def __init__(self, base_model):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(100, 10)
    def forward(self, x):
        x1 = self.base_model(x)
        return self.fc1(x1)

class BagNet(nn.Module):
    def __init__(self, patch_size):
        super(BagNet, self).__init__()
        self.patch_size = patch_size
        self.feature_model = FeatureModel()
    

    def numpy_img_to_patch(self, img):
        img1 = np.array(np.split(img, 3, axis = 1))
        img2 = np.array(np.split(img1, 3, axis = 1))
        return img2
    
    def img_to_patch(self, img):
        '''
        Args
            img - (3, W, H)
            patch_size - (Q)
        Returns
            Patches (W/Q, H/Q, 3, Q, Q)
        '''
        
        num_patches = int(img.shape[1] / self.patch_size)
        numpy_img = img.detach().permute(1, 2, 0).numpy()
        numpy_patch = self.numpy_img_to_patch(numpy_img)
        patch = torch.tensor(numpy_patch).permute(0, 1, 4, 2, 3)
        return patch
        
    
   
    def patches_to_representations(self, patches):
        '''
        Args
            patches - (W/Q, H/Q, 3, Q, Q)
        Returns
            representations - (W/Q, H/Q, V)
        '''
        representations = []
        for i in range(patches.shape[0]):
            row = []
            for j in range(patches.shape[1]):
                features = self.feature_model(patches[i,j].unsqueeze(0))
                row.append(features[0])
            representations.append(torch.stack(row))
        representations = torch.stack(representations)
        return representations

    def forward(self, x):
        batch_patches = torch.stack([self.img_to_patch(img) for img in x])
        batch_representations = torch.stack([ self.patches_to_representations(patches) for patches in batch_patches])
        return batch_representations

