"""
Blob detection using various methods (traditional image processing, UNet, DINOv2).
"""

import numpy as np
import torch
import cv2
from pathlib import Path


class BlobDetector:
    """
    Main class for blob detection using various methods.
    """
    
    def __init__(self, method='traditional', model_path=None, device=None):
        """
        Initialize the blob detector.
        
        Args:
            method: Method to use ('traditional', 'unet', 'dinov2')
            model_path: Path to trained model weights (for neural network methods)
            device: Device to use ('cuda', 'cpu', 'mps')
        """
        self.method = method
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        
        # Initialize model if using neural network method
        if method in ['unet', 'dinov2']:
            self._initialize_model()
    
    def _get_device(self, device):
        """Get the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_model(self):
        """Initialize the selected neural network model."""
        if self.method == 'unet':
            from .models import UNet
            self.model = UNet(n_channels=1, n_classes=1)
            if self.model_path and Path(self.model_path).exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
        elif self.method == 'dinov2':
            from .models import DinoV2BinarySegmentation
            self.model = DinoV2BinarySegmentation('facebook/dinov2-base')
            if self.model_path and Path(self.model_path).exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
    
    def detect_blobs(self, image, spheroid_mask=None, threshold=150):
        """
        Detect blobs in the image.
        
        Args:
            image: Input image array
            spheroid_mask: Binary mask of the spheroid (optional, for traditional method)
            threshold: Threshold for traditional blob detection
            
        Returns:
            Binary mask of detected blobs
        """
        if self.method == 'traditional':
            return self._detect_blobs_traditional(image, spheroid_mask, threshold)
        elif self.method in ['unet', 'dinov2']:
            return self._detect_blobs_neural_network(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _detect_blobs_traditional(self, image, spheroid_mask, threshold):
        """
        Detect blobs using traditional image processing.
        Based on the get_blobs_processing function from the original notebook.
        """
        if spheroid_mask is None:
            # If no spheroid mask provided, use the whole image
            spheroid_mask = np.ones_like(image, dtype=bool)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        # Apply spheroid mask
        masked_image = image_gray.copy()
        masked_image[spheroid_mask == 0] = 0
        
        # Invert the image
        inverted = cv2.bitwise_not(masked_image)
        
        # Calculate average intensity within spheroid
        avg_intensity = inverted[spheroid_mask == 1].mean()
        img_intensity = np.zeros_like(inverted)
        img_intensity[spheroid_mask == 1] = avg_intensity
        
        # Correct image by subtracting average intensity
        image_corrected = inverted.astype(np.int32) - img_intensity
        image_corrected = np.clip(image_corrected, 0, 255)
        image_corrected[spheroid_mask == 0] = 0
        image_corrected = image_corrected.astype(np.uint8)
        
        # Normalize and threshold
        image_corrected = cv2.normalize(image_corrected, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        _, blob_mask = cv2.threshold(image_corrected, threshold, 255, cv2.THRESH_BINARY)
        
        return blob_mask.astype(np.uint8)
    
    def _detect_blobs_neural_network(self, image):
        """Detect blobs using neural network models."""
        # Preprocess image
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        # Normalize
        image_normalized = image_gray.astype(np.float32) / 255.0
        
        # Convert to tensor - handle different model requirements
        if self.method == 'dinov2':
            # DINOv2 expects RGB images, so we need to convert grayscale to RGB
            image_rgb = np.stack([image_normalized] * 3, axis=0)  # Convert to 3-channel
            image_tensor = torch.from_numpy(image_rgb).unsqueeze(0)  # Add batch dimension
        else:
            # UNet expects grayscale images
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
            mask = torch.sigmoid(prediction) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        
        return mask
    
    def detect_blobs_batch(self, images, spheroid_masks=None, threshold=150):
        """
        Detect blobs in a batch of images.
        
        Args:
            images: List of input image arrays
            spheroid_masks: List of spheroid masks (optional)
            threshold: Threshold for traditional blob detection
            
        Returns:
            List of binary blob masks
        """
        blob_masks = []
        for i, image in enumerate(images):
            spheroid_mask = spheroid_masks[i] if spheroid_masks is not None else None
            blob_mask = self.detect_blobs(image, spheroid_mask, threshold)
            blob_masks.append(blob_mask)
        return blob_masks
    
    def get_blob_properties(self, blob_mask):
        """
        Get properties of detected blobs.
        
        Args:
            blob_mask: Binary mask of detected blobs
            
        Returns:
            Dictionary containing blob properties
        """
        # Find contours
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blob_properties = {
            'count': len(contours),
            'areas': [],
            'centroids': [],
            'circularities': []
        }
        
        for contour in contours:
            # Area
            area = cv2.contourArea(contour)
            blob_properties['areas'].append(area)
            
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                blob_properties['centroids'].append((cx, cy))
            else:
                blob_properties['centroids'].append((0, 0))
            
            # Circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                blob_properties['circularities'].append(circularity)
            else:
                blob_properties['circularities'].append(0)
        
        return blob_properties 