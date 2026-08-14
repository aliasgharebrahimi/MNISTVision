from dataloader import train_dataloader, eval_dataloader
from neural_network import model
from optimizer import optimizer
from loss import loss_function
from tarin import train, device
from eval import eval

EPOCHS = 10

for epochs in range(EPOCHS):

    train_loss = train(model, device, train_dataloader, optimizer, loss_function)
    eval_loss = eval(model, eval_dataloader, loss_function, device)

    print(f"train loss: {train_loss:.4f}")
    print(f"eval loss: {eval_loss:.4f}")
    print(60 * "=")