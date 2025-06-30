"""
Image preprocessing utilities for spheroid analysis.
"""

import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Union, Tuple, List
import os
from pathlib import Path


class ImagePreprocessor:
    """
    Main class for preprocessing spheroid images.
    """
    
    def __init__(self, target_size=(1280, 960), 
                 normalize=True, convert_to_grayscale=True):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for resizing (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            convert_to_grayscale: Whether to convert images to grayscale
        """
        self.target_size = target_size
        self.normalize = normalize
        self.convert_to_grayscale = convert_to_grayscale
    
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load with OpenCV for better control
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            target_size: Target size (width, height). If None, uses self.target_size
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        # OpenCV expects (width, height) for resize
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def convert_grayscale(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image: Input image array (RGB)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray
    
    def normalize_image(self, image):
        """
        Normalize image pixel values to [0, 1] and optionally mean/std.
        Supports both grayscale and RGB images.
        
        Args:
            image: Input image array
        Returns:
            Normalized image
        """
        # Scale to [0, 1]
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)
        
        # Optionally apply mean/std normalization
        if hasattr(self, 'mean_std_normalize') and self.mean_std_normalize:
            if len(normalized.shape) == 2 or (len(normalized.shape) == 3 and normalized.shape[2] == 1):
                # Grayscale
                mean = [0.5]
                std = [0.5]
                normalized = (normalized - mean[0]) / std[0]
            elif len(normalized.shape) == 3 and normalized.shape[2] == 3:
                # RGB
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                normalized = (normalized - mean) / std
        return normalized
    
    def preprocess(self, image_path, mean_std_normalize=False):
        """
        Complete preprocessing pipeline for a single image.
        Args:
            image_path: Path to the image file
            mean_std_normalize: Whether to apply mean/std normalization (default: False)
        Returns:
            Preprocessed image
        """
        # Store flag for normalization
        self.mean_std_normalize = mean_std_normalize
        # Load image
        image = self.load_image(image_path)
        # Resize
        image = self.resize_image(image)
        # Convert to grayscale if requested
        if self.convert_to_grayscale:
            image = self.convert_grayscale(image)
        # Normalize if requested
        if self.normalize:
            image = self.normalize_image(image)
        return image
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of preprocessed images
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                processed_image = self.preprocess(image_path)
                processed_images.append(processed_image)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return processed_images
    
    def crop_to_roi(self, image, roi):
        """
        Crop image to region of interest.
        
        Args:
            image: Input image array
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Cropped image
        """
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def apply_mask(self, image, mask):
        """
        Apply binary mask to image (set background to 0).
        
        Args:
            image: Input image array
            mask: Binary mask array
            
        Returns:
            Masked image
        """
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        return masked_image
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Enhance image contrast.
        
        Args:
            image: Input image array
            method: Contrast enhancement method ('clahe', 'histogram_equalization')
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image.astype(np.uint8))
        elif method == 'histogram_equalization':
            enhanced = cv2.equalizeHist(image.astype(np.uint8))
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
        
        return enhanced
    
    def remove_noise(self, image, method='gaussian'):
        """
        Remove noise from image.
        
        Args:
            image: Input image array
            method: Noise removal method ('gaussian', 'median', 'bilateral')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            denoised = cv2.medianBlur(image.astype(np.uint8), 5)
        elif method == 'bilateral':
            denoised = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
        else:
            raise ValueError(f"Unknown noise removal method: {method}")
        
        return denoised 