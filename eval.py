import torch
from tarin import device

def eval(model, dataloader, loss_func, device):

    model.eval()
    model = model.to(device)

    total_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):

            pred = model(data)
            loss = loss_func(pred, target)

            total_loss += loss.item()



    return total_loss / len(dataloader)