# Imports
import torch
import wandb
import time
from dataloader import train_dataloader, eval_dataloader
from neural_network.mnistnet import model
from optimizer import optimizer
from loss import loss_function
from tarin import train, device
from eval import eval
from config import EPOCHS, LR, OPTIMIZER, BATCH_SIZE, KERNEL_SIZE, STRIDE

# Initial w&b parameters
wandb.init(
    project="MNISTVision",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "kernel_size": KERNEL_SIZE,
        "layers": 2,
        "stride": STRIDE
    }
)

# Training Loop
for epochs in range(EPOCHS):
    # Reset peak GPU memory statistics
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Getting the current time    
    start_time = time.time()

    # Train the model for one epoch
    train_loss, train_acc = train(model, device, train_dataloader, optimizer, loss_function, stride=STRIDE)

    # Evaluate the model and generate the confusion matrix on the final epoch
    if epochs == EPOCHS:
        eval_loss, eval_acc = eval(model, eval_dataloader, loss_function, device, confusion_matrix=True, stride=STRIDE)
    else:
        eval_loss, eval_acc = eval(model, eval_dataloader, loss_function, device, stride=STRIDE)

    # Record the epoch start time
    epoch_time = time.time() - start_time

    # Get the amount of VRAM used
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram = 0

    # Print the results
    print(f"train loss: {train_loss:.4f}")
    print(f"eval loss: {eval_loss:.4f}")

    print(f"train accuracy: {train_acc:.4f}")
    print(f"eval accuracy: {eval_acc:.4f}")

    print(60 * "=")

    # Next w&b parameters
    wandb.log({
        "train_loss": train_loss,
        "val_loss": eval_loss,
        "time": epoch_time,
        "vram": vram,
        "epoch": epochs + 1,
    })

# Save Parameters Models
torch.save(model.state_dict(), "mnist_model.pth")