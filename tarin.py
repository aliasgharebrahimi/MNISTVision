import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, device, dataloader, optimizer, loss_fun ):

    model.train()
    model = model.to(device)

    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_fun(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)