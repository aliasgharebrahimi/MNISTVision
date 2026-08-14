from torch.utils.data import DataLoader
from dataset import train_data, eval_data

train_dataloader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

eval_dataloader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)