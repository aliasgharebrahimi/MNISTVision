import torch.optim as optim
from neural_network import model

optimizer = optim.Adam(model.parameters(), lr=0.001)