from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.utils.data
from torchvision import datasets, transforms

def fit(model, train_loader, metric, optimizer, model_name, epochs = 5, log_every = 1000):
    for j in range(epochs):
        total = 0
        correct = 0
        losses = []
        for i, (inputs, outputs) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions, _ = model(inputs)
            loss = metric(predictions, outputs)  
            correct += sum(torch.max(predictions, -1)[1] == outputs).item()
            total += predictions.shape[0]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i%log_every == 0:
                save_model(model, model_name + '_epoch_' + str(j) + '_batch_' + str(i))
                print(i, j, np.mean(losses), correct/total)
                
def evaluate(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, outputs = data
        predictions, _ = model(inputs)
        predictions = torch.max(predictions, -1)[1]
        correct += sum(predictions == outputs).item()
        total += predictions.shape[0]
        if i%1000 == 0:
            print(i)
    print('test accuracy', correct / total)

def save_model(model, model_name):
    torch.save(model, './models/' + model_name)

def load_model(model_name):
    model = torch.load('./models/' + model_name)
    model.eval()
    return model