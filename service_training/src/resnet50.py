from service_training.src.dataset import FloodDataset

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from loguru import logger

def train(images, labels, epochs: int=10, lr: float=0.001):
    """Train ResNet50 with PyTorch
    :params images: np.array (n, h, w, c)
    :params labels: np.array (n,)

    :returns model: pytorch model
    """
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = FloodDataset(X_train, y_train)
    val_dataset = FloodDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # 2 classes
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Setting device to {device}')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images_batch, labels_batch in val_loader:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        val_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%")
    
    return model