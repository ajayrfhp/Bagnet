from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms


def load_visualize_heatmap(model_path, test_loader):
    label_map = { 0: 'T-shirt/top', 1:'Trouser', 2 : 'Pullover',
                3: 'Dress', 4 : 'Coat', 5 : 'Sandal',
                6 : 'Shirt', 7 : 'Sneaker', 8: 'Bag',
                9 : 'Ankle boot'}
    model = load_model(model_path)
    for _ in range(1):
        inputs, outputs = next(iter(test_loader))
        predictions, features = model(inputs)
        class_heatmap = model.explain.weight.data[predictions.argmax()][0]
        f, ax = plt.subplots(2, 2)
        features = features[0, 0]

        ax[0, 0].imshow(scipy.ndimage.zoom(features.detach().numpy(), 4, order = 1), cmap = 'gray')
        class_heatmap = (class_heatmap * features).detach().numpy()
        ax[0, 1].imshow(inputs[0][0].detach().numpy(), cmap = 'gray')
        ax[1, 0].imshow(scipy.ndimage.zoom(class_heatmap, 4, order = 1), cmap = 'gray')
        print(label_map[outputs.item()], label_map[predictions.argmax().item()])
        plt.show()
        break
    pass

class ExplainModel(nn.Module):
    def get_conv_layer(self, param):
        return nn.Sequential(*[nn.Conv2d(*param), nn.ReLU(), nn.BatchNorm2d(param[1])])
    
    def __init__(self):
        super(ExplainModel, self).__init__()
        self.feature_extractor = nn.Sequential(*[
            self.get_conv_layer((1, 64, 3, 1, 1)), 
            nn.MaxPool2d(2), 
            self.get_conv_layer((64, 64, 3, 1, 1)), 
            nn.MaxPool2d(2),
            self.get_conv_layer((64, 1, 3, 1, 1))    
        ])
        self.explain = nn.Conv2d(1, 10, 7)
        self.fc1 = nn.Linear(49, 10)
    
    def forward(self, x):
        x1 = self.feature_extractor(x)
        x2 = self.explain(x1)
        return x2[:,:,0,0], x1

