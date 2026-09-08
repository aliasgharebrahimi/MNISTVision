import torch
from tarin import device
from accuracy import accuracy

def eval(model, dataloader, loss_func, device):

    model.eval()
    model = model.to(device)

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):

            pred = model(data)
            loss = loss_func(pred, target)

            total_loss += loss.item()

            pred = model(data)
            loss = loss_func(pred, target)

            total_loss += loss.item()
            total_accuracy += accuracy(pred, target)

    return total_loss / len(dataloader), total_accuracy / len(dataloader)