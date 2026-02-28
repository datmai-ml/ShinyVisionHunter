import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ShinyDataset(Dataset):
    def __init__(self, root_dir, transforms=None):

        self.root_dir = root_dir
        self.transforms = transforms

        sparkle_dir = os.path.join(root_dir, 'sparkle')
        normal_dir = os.path.join(root_dir, 'normal')