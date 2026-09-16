import torch

EPOCHS = 1
BATCH_SIZE = 16
LR = 0.001
OPTIMIZER = "Adam"
KERNEL_SIZE = 3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")