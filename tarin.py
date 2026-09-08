import torch
import torch.nn.utils as utils
from accuracy import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, device, dataloader, optimizer, loss_fun ):

    model.train()
    model = model.to(device)

    total_loss = 0
    total_acc = 0

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = data.to(device), target.to(device)

        pred = model(data)

        loss = loss_fun(pred, target)
        acc = accuracy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        def grad_model():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient = param.grad.abs().mean().item()
                    print(f"{name}: {gradient:.8f}")

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)