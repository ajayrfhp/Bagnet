from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms
from helpers import visualize_dataset, fit, evaluate, get_data, load_model
from models import SimpleModel
import sys
import argparse

def visualize(train_loader, test_loader):
    visualize_dataset(train_loader)
    visualize_dataset(test_loader)

def fit_classifier(classifier, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    fit(classifier, train_loader, criterion, optimizer, epochs = 1, model_name = 'simple_model')
    evaluate(classifier, test_loader)

def load_evaluate_model(model_path, test_loader):
    model = load_model(model_path)
    evaluate(model, test_loader)

def load_visualize_heatmap():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fit", nargs = '?', default = False)
    parser.add_argument("-model", nargs = '?', default = "simple_model")
    parser.add_argument("-visualize", nargs = '?', default = False)
    parser.add_argument("-evaluate", nargs = '?')

    args = parser.parse_args()
    train_loader, test_loader = get_data()
    model = None

    if args.model:
        if args.model == 'simple_model':
            model = SimpleModel()
    if args.fit:
        fit_classifier(model, train_loader, test_loader)
    elif args.visualize:
        visualize(train_loader, test_loader)
    elif args.evaluate:
        load_evaluate_model(args.evaluate, test_loader)

main()
