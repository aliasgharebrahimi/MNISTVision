# Imports
from torch.utils.data import DataLoader
from dataset import train_data, eval_data
from config import BATCH_SIZE

# Training Dataloader
train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# Evaling Dataloader
eval_dataloader = DataLoader(
    eval_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)