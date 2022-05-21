import torch

def accuracy(y_pred, y):
    n_accurate = torch.sum(y_pred.argmax(axis=1) == y)

    return n_accurate / len(y)