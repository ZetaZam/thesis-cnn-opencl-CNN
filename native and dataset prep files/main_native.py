import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os

# ========== CONFIGURATION ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PYTORCH_DEVICE = "cuda"  # or "cpu"
DEVICE = torch.device(PYTORCH_DEVICE if torch.cuda.is_available() and PYTORCH_DEVICE=="cuda" else "cpu")

WEIGHTS_PATH = "alexnet_weights_pretrained.npz"  # set None to skip loading
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.1
SEED = 42

start_time = time.perf_counter()
print(f"Start time: {start_time:.2f} seconds")
print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Validation split: {VALIDATION_SPLIT}")
print(f"Seed: {SEED}")
print(f"Using weights from: {WEIGHTS_PATH}")


# ========== MODEL DEFINITION ==========
# Adapted AlexNet for CIFAR-10 (input 3x32x32)
class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3 for CIFAR-10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========== LOAD WEIGHTS FROM NPZ ==========
def load_weights_npz(model, npz_path):
    print(f"Loading weights from {npz_path}")
    weights = np.load(npz_path)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                w = weights[name]
                if param.shape == w.shape:
                    param.copy_(torch.tensor(w))
                else:
                    print(f"Shape mismatch for {name}, skipping weight load")
            else:
                print(f"Weight {name} not found in npz, skipping")
    print("Weight loading done.")

# ========== DATASET PREP ==========
def prepare_dataloaders(batch_size, validation_split, seed):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    dataset_size = len(full_trainset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    torch.manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

    # Use different transform for val dataset
    val_dataset.dataset.transform = transform_val

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader

# ========== TRAINING & VALIDATION ==========
def train_one_epoch(model, device, trainloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    start = time.time()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    end = time.time()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch {epoch} training time: {end - start:.2f}s, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

def validate(model, device, valloader, criterion):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= total
    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc

# ========== MAIN ==========
def main():
    print(f"Using device: {DEVICE}")
    model = AlexNetCIFAR10().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Optionally load weights
    if WEIGHTS_PATH:
        try:
            load_weights_npz(model, WEIGHTS_PATH)
        except Exception as e:
            print(f"Failed to load weights: {e}")

    trainloader, valloader = prepare_dataloaders(BATCH_SIZE, VALIDATION_SPLIT, SEED)

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, DEVICE, trainloader, criterion, optimizer, epoch)
        validate(model, DEVICE, valloader, criterion)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Device info
    if DEVICE.type == 'cuda':
        device_name = torch.cuda.get_device_name(DEVICE)
        props = torch.cuda.get_device_properties(DEVICE)
        compute_units = props.multi_processor_count  # Number of SMs (Streaming Multiprocessors)
    elif DEVICE.type == 'cpu':
        device_name = "CPU"
        # Optional: number of physical cores
        try:
            import psutil
            compute_units = psutil.cpu_count(logical=False)
        except ImportError:
            compute_units = os.cpu_count()  # fallback to logical cores
    else:
        device_name = str(DEVICE)
        compute_units = "N/A"

    print(f"Device: {device_name}")
    print(f"Compute units: {compute_units}")
    print(f"Training time elapsed: {elapsed_time:.3f} seconds")





if __name__ == "__main__":
    main()
