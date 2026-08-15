from torch.utils.data import DataLoader
from dataset import train_data, eval_data

train_dataloader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

eval_dataloader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)