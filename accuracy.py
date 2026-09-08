import torch

def accuracy(pred, labels):
    pred = torch.argmax(dim=1)

    correct = (pred == labels).sum()
    total = len(labels)
    return correct / total