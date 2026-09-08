def accuracy(pred, labels):
    correct = (pred == labels).sum()
    total = len(labels)
    return correct / total