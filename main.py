from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms
from helpers import visualize_dataset, fit, evaluate, load_model
import fashion_mnist_dataset
import dogs_cats_dataset
from models import SimpleModel, ExplainModel
import sys
import argparse
import scipy.ndimage

def visualize(train_loader, test_loader):
    visualize_dataset(train_loader)
    visualize_dataset(test_loader)

def fit_classifier(classifier, train_loader, test_loader, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    fit(classifier, train_loader, criterion, optimizer, epochs = 1, model_name = model_name)
    evaluate(classifier, test_loader)

def load_evaluate_model(model_path, test_loader):
    model = load_model(model_path)
    evaluate(model, test_loader)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fit", nargs = '?', default = False)
    parser.add_argument("-model", nargs = '?', default = "simple_model")
    parser.add_argument("-visualize", nargs = '?', default = False)
    parser.add_argument("-evaluate", nargs = '?')
    parser.add_argument("-model_path", nargs = '?')
    parser.add_argument("-visualize_heatmap", nargs = '?')


    args = parser.parse_args()
    train_loader, test_loader = get_fashion_mnist_data()
    model = None

    if args.model:
        if args.model == 'simple_model':
            model = SimpleModel()
        if args.model == 'explain_model':
            model = ExplainModel()
        
    if args.fit:
        fit_classifier(model, train_loader, test_loader, args.model)
    elif args.visualize:
        visualize(train_loader, test_loader)
    elif args.evaluate and args.model_path:
        load_evaluate_model(args.model_path, test_loader)
    elif args.visualize_heatmap and args.model_path:
        load_visualize_heatmap(args.model_path, test_loader)

main()
