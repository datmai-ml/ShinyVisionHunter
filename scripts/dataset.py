import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ShinyDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        sparkle_dir = os.path.join(root_dir, 'sparkle')
        normal_dir = os.path.join(root_dir, 'normal')


        # Load shinies
        if os.path.exists(sparkle_dir):
            for batch_folder in sorted(os.listdir(sparkle_dir)):
                batch_path = os.path.join(sparkle_dir, batch_folder)
                if os.path.isdir(batch_path):
                    self.samples.append((batch_path, 1)) # Shinies = label 1
        
        # Load normal
        if os.path.exists(normal_dir):
            for batch_folder in sorted(os.listdir(normal_dir)):
                batch_path = os.path.join(normal_dir, batch_folder)
                if os.path.isdir(batch_path):
                    self.samples.append((batch_path, 0)) # Shinies = label 1

        print(f"Loaded {len(self.samples)} batches from {root_dir}")


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        batch_path, label = self.samples[idx]

        frame_files = sorted([f for f in os.listdir(batch_path)])

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(batch_path, frame_file)
            image = Image.open(frame_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)
        label = torch.tensor(label, dtype=torch.float32)

        return frames, label