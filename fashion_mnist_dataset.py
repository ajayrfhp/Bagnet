import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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