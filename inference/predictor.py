import torch
from neural_network import MNISTNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))

model.to(device)
model.eval()