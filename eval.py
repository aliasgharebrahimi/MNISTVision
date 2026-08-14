import torch
from tarin import device

def eval(model, data, dataloader, loss_func, device):

    model.eval()
    model = model.to(device)

    total_loss = 0

    with torch.norm():

        pred = model(data)
        loss = loss_func(pred, data)

        total_loss += loss.item()

    return total_loss / len(dataloader)