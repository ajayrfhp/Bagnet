import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import glob

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../data', train=True, download=True, transform = transform),
            batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../data', train=False, download=True, transform = transform),
            batch_size=1, shuffle=True)
    return train_loader, test_loader

def visualize_dataset(data_loader, num = 1):
    label_map = { 0: 'T-shirt/top', 1:'Trouser', 2 : 'Pullover',
                3: 'Dress', 4 : 'Coat', 5 : 'Sandal',
                6 : 'Shirt', 7 : 'Sneaker', 8: 'Bag',
                9 : 'Ankle boot'}
    for _ in range(num):
        inputs, outputs = next(iter(data_loader))
        plt.figure(figsize=(3,3))
        plt.title(label_map[outputs[0].item()])
        plt.imshow(inputs[0][0], cmap = 'gray')
        plt.show()
        