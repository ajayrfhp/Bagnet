import argparse
from explain_model import ExplainModel, load_visualize_heatmap
import fashion_mnist_dataset
import dogs_cats_dataset
from helpers import fit, evaluate, load_model
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage
from simple_model import SimpleModel
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fit", nargs = '?', default = False)
    parser.add_argument("-model", nargs = '?', default = "simple_model")
    parser.add_argument("-visualize", nargs = '?', default = False)
    parser.add_argument("-evaluate", nargs = '?')
    parser.add_argument("-dataset")
    parser.add_argument("-model_path", nargs = '?')
    parser.add_argument("-visualize_heatmap", nargs = '?')


    args = parser.parse_args()
    if args.dataset:
        if args.dataset == 'fashion_mnist':
            train_loader, test_loader = fashion_mnist_dataset.get_data_loaders()
            visualize = fashion_mnist_dataset.visualize_dataset
        elif args.dataset == "dogs_cats":
            train_loader, test_loader = dogs_cats_dataset.get_data_loaders()
            visualize = dogs_cats_dataset.visualize_dataset
    if args.model:
        if args.model == 'simple_model':
            model = SimpleModel()
        if args.model == 'explain_model':
            model = ExplainModel()
    if args.fit:
        fit_classifier(model, train_loader, test_loader, args.model)
    elif args.visualize:
        visualize(train_loader)
    elif args.evaluate and args.model_path:
        model = load_model(model_path)
        evaluate(model, test_loader)
    elif args.visualize_heatmap and args.model_path:
        load_visualize_heatmap(args.model_path, test_loader)

main()
