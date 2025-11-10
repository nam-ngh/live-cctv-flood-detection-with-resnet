from service_training.config import (
    S, 
    FLOOD_IMAGES_PATH, 
    DRY_IMAGES_PATH, 
    EPOCHS, 
    MODEL_PATH,
)
from service_training.src import preprocessing, resnet50

import numpy as np
import torch

def load_data():
    # load images and label
    flood_images = preprocessing.load_raw_images(FLOOD_IMAGES_PATH,)
    dry_images = preprocessing.load_raw_images(DRY_IMAGES_PATH,)
    images = np.concatenate([flood_images, dry_images], axis=0)

    flood_labels = np.ones((flood_images.shape[0],))
    dry_labels = np.zeros((dry_images.shape[0],))
    labels = np.concatenate([flood_labels, dry_labels], axis=0)

    # random shuffle with seed
    np.random.seed(42)
    shuffled_indices = np.arange(images.shape[0])
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels

def main():
    images, labels = load_data()
    model = resnet50.train(images, labels, epochs=EPOCHS)
    save_path = MODEL_PATH
    torch.save(model.state_dict(), save_path)
    S.logger.success(f'Saved trained model to {save_path}')
if __name__ == '__main__':
    main()