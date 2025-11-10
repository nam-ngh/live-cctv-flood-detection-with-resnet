from service_training.config import (
    S, 
    FLOOD_IMAGES_PATH, 
    DRY_IMAGES_PATH,
    MODEL_PATH,
)
from service_training.src import preprocessing, resnet50
from service_inference.src import predict as prd

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions_grid(images, predictions, confidences, probabilities):
    """
    Plot 6 images in 3x2 grid with prediction results
    
    Args:
        images: numpy array (6, H, W, 3)
        predictions: numpy array (6,) with values 0 or 1
        confidences: numpy array (6,) with confidence values
        probabilities: numpy array (6, 2) with [no_flood_prob, flood_prob]
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        
        # Display image
        ax.imshow(images[i])
        ax.axis('off')
        
        # Get prediction info
        pred = predictions[i]
        conf = confidences[i]
        flood_prob = probabilities[i]
        
        # Determine color and label
        if pred == 1:
            color = 'red'
            label = '🚨 FLOOD'
            bg_color = 'darkred'
        else:
            color = 'green'
            label = '✅ NO FLOOD'
            bg_color = 'darkgreen'
        
        # Create text box with results
        textstr = f'{label}\n'
        textstr += f'Conf: {conf:.1%}\n'
        textstr += f'Flood: {flood_prob:.1%}\n'
        
        # Add text box overlay
        props = dict(boxstyle='round', facecolor=bg_color, alpha=0.85, pad=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=props, color='white', fontweight='bold')
        
        # Add image number
        ax.text(0.95, 0.05, f'#{i+1}', transform=ax.transAxes,
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, pad=0.3))
    
    # Add overall title with timestamp
    # fig.suptitle(f'Flood Detection Results - 🕒 {timestamp}', 
    #              fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    plt.show()


def main():
    # Load images and labels
    flood_images = preprocessing.load_raw_images(FLOOD_IMAGES_PATH, max_num_images=10)
    dry_images = preprocessing.load_raw_images(DRY_IMAGES_PATH, max_num_images=10)
    images = np.concatenate([flood_images, dry_images], axis=0)
    
    # Shuffle
    np.random.seed(42)
    shuffled_indices = np.arange(images.shape[0])
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    
    # Load model
    model = prd.load_flood_model(model_path=MODEL_PATH)
    
    # Predict on first 6 images
    predictions = []
    confidences = []
    probabilities = []
    
    for i in range(6):
        pred, conf, prob = prd.predict_flood(
            model=model,
            frame=images[i],
        )
        predictions.append(pred)
        confidences.append(conf)
        probabilities.append(prob)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Plot results
    plot_predictions_grid(
        images[:6], 
        predictions, 
        confidences, 
        probabilities
    )

if __name__ == '__main__':
    main()