import torch
import wandb
import time
from dataloader import train_dataloader, eval_dataloader
from neural_network import model
from optimizer import optimizer
from loss import loss_function
from tarin import train, device
from eval import eval
from config import EPOCHS, LR, OPTIMIZER, BATCH_SIZE, KERNEL_SIZE

wandb.init(
    project="MNISTVision",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "kernel_size": KERNEL_SIZE,
        "layers": 2
    }
)

for epochs in range(EPOCHS):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    train_loss = train(model, device, train_dataloader, optimizer, loss_function)
    eval_loss = eval(model, eval_dataloader, loss_function, device)

    epoch_time = time.time() - start_time
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram = 0

    print(f"train loss: {train_loss:.4f}")
    print(f"eval loss: {eval_loss:.4f}")
    print(60 * "=")

    wandb.log({
        "train_loss": train_loss,
        "val_loss": eval_loss,
        "time": epoch_time,
        "vram": vram,
        "epoch": epochs + 1,
    })

torch.save(model.state_dict(), "mnist_model.pth")