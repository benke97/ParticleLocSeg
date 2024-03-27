import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass
import numpy as np

def preprocess_image(image_path):
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
    normalized_image = normalized_image[np.newaxis,np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    return image_tensor

def center_of_mass_positions(predicted_localization,binary_threshold=0.1):
    # Create a binary mask for the predicted localization
    predicted_localization = predicted_localization.cpu().squeeze().numpy()
    binary_segmentation = predicted_localization >= binary_threshold
    labeled_image = label(binary_segmentation)
    regions = regionprops(labeled_image)
    centers_of_mass = []
    for region in regions:
        # Create a binary mask for this region
        mask = (labeled_image == region.label)
        # Extract the region's pixel values from the grayscale image
        region_values = mask * predicted_localization
        
        # Compute the center of mass for the region using the grayscale values
        center = center_of_mass(region_values)
        centers_of_mass.append(center)
    
    return np.array(centers_of_mass)