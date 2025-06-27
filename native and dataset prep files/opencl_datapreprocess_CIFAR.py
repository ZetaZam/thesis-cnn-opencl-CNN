import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os

# Parameters
output_dir = "./cifar10_resized"
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

transform = Compose([
    Resize((227, 227)),
    ToTensor()
])

# Download dataset
dataset = CIFAR10(root="./data", train=False, download=True)

# Process and save the first 100 samples
for idx, (img, label) in enumerate(dataset):
    if idx >= 100:
        break
    img_resized = transform(img)  # Tensor shape [C, H, W]
    img_npy = img_resized.numpy()
    np.save(f"{output_dir}/images/img_{idx:03d}.npy", img_npy)
    with open(f"{output_dir}/labels/labels.npy", 'ab') as f:
        np.save(f, np.array([label]))

print(" Saved 100 resized CIFAR-10 images and labels as .npy files.")
