import os
import tarfile
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

def load_datasets(root_dir='caltech-101',batch_size=32):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    # Data preprocessing with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract the tar.gz file if it hasn't been extracted yet
    extracted_dir = os.path.join(root_dir, '101_ObjectCategories')
    tar_path = os.path.join(root_dir, '101_ObjectCategories.tar.gz')

    if not os.path.exists(extracted_dir):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=root_dir)
    else:
        print(f"{extracted_dir} already exists. Skipping extraction.")

    # Load the dataset using ImageFolder
    train_dataset = datasets.ImageFolder(root=extracted_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=extracted_dir, transform=val_transform)

    # Split dataset into train and validation sets (using standard split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
