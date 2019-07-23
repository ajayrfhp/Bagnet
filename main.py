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
from models import Classifier, FeatureModel
import sys
import argparse

def visualize(train_loader, test_loader):
    visualize_dataset(train_loader)
    visualize_dataset(test_loader)

def fit_feature_model(train_loader, test_loader):
    feature_model = FeatureModel()
    classifier = Classifier(feature_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    fit(classifier, train_loader, criterion, optimizer, epochs = 1, model_name = 'feature_model')
    evaluate(classifier, test_loader)

def fit_bagnet_model(train_loader, test_loader):
    pass

def load_evaluate_model(model_path, test_loader):
    model = load_model(model_path)
    evaluate(model, test_loader)

def load_visualize_heatmap():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fit", nargs = '?', default = False)
    parser.add_argument("-model", nargs = '?', default = "feature_model")
    parser.add_argument("-visualize", nargs = '?', default = False)
    parser.add_argument("-evaluate", nargs = '?')

    args = parser.parse_args()
    train_loader, test_loader = get_data()
    print(args)
    if args.fit and args.model == "feature_model":
        fit_feature_model(train_loader, test_loader)
    elif args.fit and args.model == 'bagnet':
        fit_bagnet_model(train_loader, test_loader)
    elif args.visualize:
        visualize(train_loader, test_loader)
    elif args.evaluate:
        load_evaluate_model(args.evaluate, test_loader)

main()
