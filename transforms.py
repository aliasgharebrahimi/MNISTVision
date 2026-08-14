import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

eval_transforms = transforms.Compose([
    transforms.ToTensor()
])