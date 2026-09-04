import torch
from config import SEED

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)