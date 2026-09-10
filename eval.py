import torch
from tarin import device
from accuracy import accuracy
from confusion_matrix import confusion_matrix, plot_confusion_matrix

def eval(model, dataloader, loss_func, device):

    model.eval()
    model = model.to(device)

    total_loss = 0
    total_accuracy = 0

    all_pred = []
    all_label = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):

            pred = model(data)
            loss = loss_func(pred, target)

            all_pred.append(pred)
            all_label.append(target)

            total_loss += loss.item()
            total_accuracy += accuracy(pred, target)

    all_pred = torch.cat(all_pred, dim=0)
    all_label = torch.cat(all_label, dim=0)

    cm = confusion_matrix(all_pred, all_label, 10)
    plot_confusion_matrix(cm)

    return total_loss / len(dataloader), total_accuracy / len(dataloader)