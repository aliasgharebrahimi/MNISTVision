# Imports
import torch
from config import SEED

# Set the random seed for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)