import cv2
import numpy as np
import torch


def preprocess(canvas):
    # Convert black digit on white background
    # to white digit on black background
    image = 255 - canvas

    # Find the digit
    _, binary = cv2.threshold(
        image,
        30,
        255,
        cv2.THRESH_BINARY
    )

    # Find bounding box of the digit
    coords = cv2.findNonZero(binary)

    # If canvas is empty
    if coords is None:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Crop only the digit
    digit = image[y:y + h, x:x + w]

    # Make the digit area square while preserving its shape
    size = max(w, h)

    pad = int(size * 0.2)

    digit = cv2.copyMakeBorder(
        digit,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize the digit while keeping it smaller than 28x28
    digit = cv2.resize(
        digit,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    # Create MNIST-like 28x28 image
    result = np.zeros((28, 28), dtype=np.uint8)

    # Center the digit
    start = (28 - 20) // 2

    result[
        start:start + 20,
        start:start + 20
    ] = digit

    # Convert to tensor
    tensor = torch.tensor(
        result,
        dtype=torch.float32
    ) / 255.0

    # [28, 28] -> [1, 1, 28, 28]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor