"""
Feature extraction for spheroid analysis.
"""

import numpy as np
import cv2
from ..utils.metrics import (
    calculate_area, calculate_perimeter, calculate_radius, 
    calculate_compactness, calculate_eccentricity
)


class FeatureExtractor:
    """
    Extract features from spheroid images and masks.
    """
    
    def __init__(self, pixel_size=1.0):
        """
        Initialize the feature extractor.
        
        Args:
            pixel_size: Size of each pixel in micrometers
        """
        self.pixel_size = pixel_size
    
    def extract_spheroid_features(self, spheroid_mask):
        """
        Extract features from spheroid mask.
        
        Args:
            spheroid_mask: Binary mask of the spheroid
            
        Returns:
            Dictionary of spheroid features
        """
        features = {}
        
        # Basic geometric features
        features['spheroid_area'] = calculate_area(spheroid_mask, self.pixel_size)
        features['spheroid_perimeter'] = calculate_perimeter(spheroid_mask, self.pixel_size)
        features['spheroid_radius'] = calculate_radius(spheroid_mask, self.pixel_size)
        features['spheroid_circularity'] = calculate_compactness(spheroid_mask)
        
        return features
    
    def extract_blob_features(self, image, spheroid_mask, blob_mask=None):
        """
        Extract blob-related features from image within spheroid region.
        
        Args:
            image: Input image (grayscale)
            spheroid_mask: Binary mask of the spheroid
            blob_mask: Binary mask of detected blobs (optional, will detect if not provided)
            
        Returns:
            Dictionary of blob features
        """
        # Use provided blob mask or detect blobs
        if blob_mask is None:
            # Apply spheroid mask to image
            masked_image = image.copy()
            masked_image[spheroid_mask == 0] = 0
            
            # Detect blobs
            blob_mask = self._detect_blobs(masked_image)
        
        features = {}
        
        # Blob count and properties
        features['blob_count'] = self._count_blobs(blob_mask)
        features['blob_area_total'] = calculate_area(blob_mask, self.pixel_size)
        features['blob_area_relative'] = features['blob_area_total'] / calculate_area(spheroid_mask, self.pixel_size)
        
        # Blob distribution features
        if features['blob_count'] > 0:
            features['blob_density'] = features['blob_count'] / calculate_area(spheroid_mask, self.pixel_size)
            features['blob_size_mean'] = features['blob_area_total'] / features['blob_count']
        else:
            features['blob_density'] = 0.0
            features['blob_size_mean'] = 0.0
        
        return features
    
    def _detect_blobs(self, image):
        """
        Detect blobs in the image using intensity-based thresholding.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary mask of detected blobs
        """
        # Invert image (blobs are typically dark)
        inverted = cv2.bitwise_not(image)
        
        # Calculate average intensity within non-zero regions
        non_zero_pixels = inverted[image > 0]
        if len(non_zero_pixels) > 0:
            avg_intensity = non_zero_pixels.mean()
        else:
            avg_intensity = 0
        
        # Create intensity-corrected image
        img_intensity = np.zeros_like(inverted)
        img_intensity[image > 0] = avg_intensity
        
        # Correct image
        image_corrected = inverted.astype(np.int32) - img_intensity
        image_corrected = np.clip(image_corrected, 0, 255)
        image_corrected = image_corrected.astype(np.uint8)
        
        # Apply mask
        image_corrected[image == 0] = 0
        
        # Normalize and threshold
        image_corrected = cv2.normalize(image_corrected, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        _, blob_mask = cv2.threshold(image_corrected, 100, 255, cv2.THRESH_BINARY)
        
        return blob_mask
    
    def _count_blobs(self, blob_mask):
        """
        Count the number of connected components (blobs) in the mask.
        
        Args:
            blob_mask: Binary mask of blobs
            
        Returns:
            Number of blobs
        """
        num_labels, labels = cv2.connectedComponents(blob_mask)
        return num_labels - 1  # Subtract 1 for background
    

    
    def extract_all_features(self, image, spheroid_mask, blob_mask=None):
        """
        Extract all features from image, spheroid mask, and blob mask.
        
        Args:
            image: Input grayscale image
            spheroid_mask: Binary mask of the spheroid
            blob_mask: Binary mask of detected blobs (optional)
            
        Returns:
            Dictionary containing all extracted features
        """
        spheroid_features = self.extract_spheroid_features(spheroid_mask)
        blob_features = self.extract_blob_features(image, spheroid_mask, blob_mask)
        
        # Combine all features
        all_features = {**spheroid_features, **blob_features}
        
        return all_features
    
    def extract_features_batch(self, images, spheroid_masks):
        """
        Extract features from a batch of images and masks.
        
        Args:
            images: List of input images
            spheroid_masks: List of spheroid masks
            
        Returns:
            List of feature dictionaries
        """
        if len(images) != len(spheroid_masks):
            raise ValueError("Number of images must match number of masks")
        
        features_list = []
        for image, mask in zip(images, spheroid_masks):
            features = self.extract_all_features(image, mask)
            features_list.append(features)
        
        return features_list 