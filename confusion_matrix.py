import torch
import matplotlib.pyplot as plt

def confusion_matrix(pred, labels, num_classes):
    pred = torch.argmax(pred, dim=1)

    cm = torch.zeros(num_classes, num_classes)

    for true, predicted in zip(labels, pred):
        cm[true, predicted] += 1

def plot_confusion_matrix(cm):

    plt.figure(figsize=(8, 8))

    plt.imshow(cm, cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.colorbar()

    plt.xticks(range(3))
    plt.yticks(range(3))

    plt.show()