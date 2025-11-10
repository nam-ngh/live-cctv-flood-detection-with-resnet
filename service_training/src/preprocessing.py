import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

def plot_4x4(images: np.ndarray,):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.flatten()

    # loop through your images and plot each one
    for i, img in enumerate(images):
        ax = axs[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

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

def load_raw_images(zip_path, target_size=(224, 224), max_num_images: int=1000):
    """
    Load images from zip file, convert to numpy and crop+resize
    
    :params zip_path: Path to zip file
    :params target_size: (height, width)
    
    :returns images: numpy array (N, H, W, 3)
    """
    images = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # get all image files
        image_files = [
            f for f in zip_ref.namelist() 
            if f.lower().endswith(('.jpg')) 
            and not f.startswith('__MACOSX')
        ]
        logger.info(
            f'{len(image_files)} images available, starting to load {max_num_images}'
        )
        err_count = 0
        seen_hashes = set()

        # loop through selected file paths and process imgs
        for file_path in tqdm(
                image_files[:max_num_images], desc="Loading images"
        ):
            try:
                # Read image from zip
                image_data = zip_ref.read(file_path)
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue

                # cropping to square
                if img.shape[0] != img.shape[1]:
                    img = crop_square_centre(img)
                
                # hashing to check dups
                img_hash = hash(img.tobytes())
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    # standard resize + convert color scheme
                    original_size = img.shape
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    logger.debug(f'Converted {file_path} img from {original_size} to {img.shape}')
                else:
                    logger.debug(f'Image duplicate found, skipping!')
                
            except Exception as e:
                logger.debug(f'Error loading image from path {file_path}: {e}')
                err_count += 1
                continue
    
    output = np.stack(images, axis=0)
    logger.info(
        f'Finished loading {len(images)} images with {err_count} errors. Output array shape: {output.shape}'
    )
    return output