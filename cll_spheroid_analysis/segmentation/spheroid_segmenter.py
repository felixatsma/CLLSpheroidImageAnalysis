"""
Spheroid segmentation using SAM2 for spheroid masking.
"""

import numpy as np
import torch
import cv2
from pathlib import Path

# Try to import SAM2 (optional dependency)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Install with: pip install segment-anything-2")


class SpheroidSegmenter:
    """
    Main class for spheroid segmentation using SAM2.
    """
    
    def __init__(self, sam2_checkpoint=None, device=None):
        """
        Initialize the spheroid segmenter.
        
        Args:
            sam2_checkpoint: Path to SAM2 checkpoint (required)
            device: Device to use ('cuda', 'cpu', 'mps')
        """
        self.sam2_checkpoint = sam2_checkpoint
        self.device = self._get_device(device)
        self.sam2_predictor = None
        
        # Initialize SAM2
        if sam2_checkpoint:
            self._initialize_sam2()
    
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
    
    def _initialize_sam2(self):
        """Initialize SAM2 model."""
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not available. Please install it first.")
        if not self.sam2_checkpoint:
            raise ValueError("SAM2 checkpoint path is required.")
        
        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            self.sam2_checkpoint,
            device=self.device,
            apply_postprocessing=False
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2)
    
    def segment_spheroid(self, image, center_point=None):
        """
        Segment spheroid from image using SAM2.
        
        Args:
            image: Input image array
            center_point: Center point for SAM2 (if None, uses image center)
            
        Returns:
            Binary mask of the spheroid
        """
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 predictor not initialized. Please provide sam2_checkpoint.")
        
        # Set image
        self.sam2_predictor.set_image(image)
        
        # Use center point or image center
        if center_point is None:
            center_point = (image.shape[1] // 2, image.shape[0] // 2)
        
        input_point = np.array([center_point])
        input_label = np.array([1])
        
        # Predict
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        return masks[0].astype(np.uint8)
    
    def segment_batch(self, images, center_points=None):
        """
        Segment a batch of images.
        
        Args:
            images: List of input image arrays
            center_points: List of center points (optional)
            
        Returns:
            List of binary masks
        """
        masks = []
        for i, image in enumerate(images):
            center_point = center_points[i] if center_points is not None else None
            mask = self.segment_spheroid(image, center_point)
            masks.append(mask)
        return masks
    
    def get_bounding_box(self, mask):
        """
        Get bounding box of a binary mask.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Bounding box as (x_min, x_max, y_min, y_max)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return xmin, xmax, ymin, ymax
    
    def crop_to_mask(self, image, mask, margin=10):
        """
        Crop image to the region of the mask with margin.
        
        Args:
            image: Input image array
            mask: Binary mask array
            margin: Margin around the mask in pixels
            
        Returns:
            Cropped image
        """
        xmin, xmax, ymin, ymax = self.get_bounding_box(mask)
        
        # Add margin
        xmin = max(0, xmin - margin)
        xmax = min(image.shape[1], xmax + margin)
        ymin = max(0, ymin - margin)
        ymax = min(image.shape[0], ymax + margin)
        
        cropped = image[ymin:ymax, xmin:xmax]
        return cropped 