import numpy as np
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    """PyTorch Dataset for flood images"""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array (N, H, W, 3) - RGB, 0-255
            labels: numpy array (N,)
            transform: optional transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: normalize for ResNet
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            # Convert to CHW format for PyTorch
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label