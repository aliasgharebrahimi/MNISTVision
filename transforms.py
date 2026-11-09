# Imports
import torchvision.transforms as transforms

# Training Transforms
train_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Evaling Transforms
eval_transforms = transforms.Compose([
    transforms.ToTensor()
])