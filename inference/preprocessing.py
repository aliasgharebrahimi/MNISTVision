import cv2
import torch


def preprocess(canvas):
    image = cv2.resize(canvas, (28, 28))

    image = 255 - image

    image = torch.tensor(image, dtype=torch.float32) / 255.0

    image = image.unsqueeze(0)
    image = image.unsqueeze(0)

    return image