import torch
import matplotlib.pyplot as plt

def confusion_matrix(pred, labels, num_classes):
    pred = torch.argmax(pred, dim=1)

    cm = torch.zeros(num_classes, num_classes)

    for true, predicted in zip(labels, pred):
        cm[true, predicted] += 1