# Imports
import torch.optim as optim
from neural_network import model
from config import LR

# Initialize the Adam optimizer  the learning rate
optimizer = optim.Adam(model.parameters(), lr=LR)