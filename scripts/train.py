import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import ShinyDetector
from dataset import ShinyDataset

def train():
    # Configurations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


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
    loss_func = nn.BCELoss()
    model = ShinyDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)
            labels = labels.to(device)
            train_outputs = model(frames)
            loss = loss_func(train_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                val_outputs = model(frames)
                loss = loss_func(val_outputs, labels)
                total_val_loss += loss.item()

                predictions = (val_outputs > 0.5).float() # probability of shiny. if model guesses > 0.5, then shiny.
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), 'shiny_detector.pth')
    print("Training complete! Model saved to shiny_detector.pth")

if __name__ == '__main__':
    train()
                                                                                 