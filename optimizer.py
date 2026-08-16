import torch.optim as optim
from neural_network import model
from config import LR

optimizer = optim.Adam(model.parameters(), lr=LR)