import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import ShinyDetector
from dataset import ShinyDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ShinyDataset(root_dir='data/train', transform=transform)
val_dataset = ShinyDataset(root_dir='data/val', transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

lr = 0.001
epochs = 10
loss = nn.BCELoss()
model = ShinyDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for batch                                                                                                                                                                                                                            
                                                                                 