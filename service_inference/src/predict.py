import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
from loguru import logger
import matplotlib.pyplot as plt
from datetime import datetime

class ModelNotFoundError(Exception):
    pass

def load_flood_model(model_path, device='cuda'):
    try:
        model = resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise ModelNotFoundError(f'Model failed to load: {e}')

def crop_square_centre(img):
    if img.shape[0] < img.shape[1]:
        start = int((img.shape[1] - img.shape[0])/2)
        end = start + img.shape[0]
        return img[:, start:end, :]
    elif img.shape[1] < img.shape[0]:
        start = int((img.shape[0] - img.shape[1])/2)
        end = start + img.shape[1]
        return img[start:end, :, :]
    else:
        logger.info(f'Cropping skipped for image sized {img.shape}')
        return img

def preprocess_frame(frame):
    """
    Preprocess CCTV frame for ResNet50

    :params frame: numpy array (H, W, 3) - RGB, 0-255
    
    :returns tensor: torch.Tensor (1, 3, H, W)
    """
    # resize, scale
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # tensor conversion
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    # batch
    frame_tensor = frame_tensor.unsqueeze(0)
    return frame_tensor

def predict_flood(model, frame, device='cuda'):
    """
    Predict if frame contains flooding
    
    :params model: trained PyTorch model
    :params frame: numpy array (H, W, 3) - RGB, 0-255
    :params device: 'cpu' or 'cuda'
    
    :returns prediction: int (0=no flood, 1=flood)
    :returns confidence: float (0-1)
    :returns probabilities: dict with both class probabilities
    """
    # Preprocess
    frame_tensor = preprocess_frame(frame)
    frame_tensor = frame_tensor.to(device)
    
    # Predict
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(frame_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 0)  # Get max probability
    
    # Convert to Python types
    prediction = predicted.item()
    confidence = confidence.item()
    probs = {
        'no_flood': probabilities[0].item(),
        'flood': probabilities[1].item()
    }
    
    return prediction, confidence, probs['flood']

def plot_prediction(frame, prediction, confidence, probs):
    """
    Plot frame with prediction results
    
    Args:
        frame: numpy array (H, W, 3) RGB 0-255
        prediction: 0 or 1
        confidence: float 0-1
        probs: dict with 'flood' and 'no_flood' probabilities
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.axis('off')
    
    # Determine color based on prediction
    color = 'red' if prediction == 1 else 'green'
    label = 'WATER POOLING DETECTED' if prediction == 1 else 'NO FLOOD'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add prediction text
    plt.title(
        f"{label}\nConfidence: {confidence:.1%} | Probability: {probs:.1%} | Time: {timestamp}",
        fontsize=20, fontweight='bold', color=color, pad=20,
    )
    
    plt.tight_layout()
    plt.show()