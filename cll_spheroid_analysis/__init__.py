"""
CLL Spheroid Image Analysis Package

A comprehensive Python package for analyzing images of spheroids of leukemia cells.
Provides preprocessing, segmentation, feature extraction, and prediction capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .pipeline import SpheroidAnalysisPipeline
from .preprocessing import ImagePreprocessor, ImageAugmenter
from .segmentation import (
    SpheroidSegmenter, BlobDetector, UNet, DinoV2BinarySegmentation, 
    SegmentationDataset, SegmentationTrainer, create_dataloaders, train_segmentation_model
)
from .feature_extraction import FeatureExtractor

# Import utility functions
from .utils.visualization import show_mask, show_mask_overlay, plot_results
from .utils.metrics import dice_score, iou_score, calculate_circularity

__all__ = [
    "SpheroidAnalysisPipeline",
    "ImagePreprocessor",
    "ImageAugmenter", 
    "SpheroidSegmenter",
    "BlobDetector",
    "UNet",
    "DinoV2BinarySegmentation",
    "SegmentationDataset",
    "SegmentationTrainer",
    "create_dataloaders",
    "train_segmentation_model",
    "FeatureExtractor",
    "show_mask",
    "show_mask_overlay", 
    "plot_results",
    "dice_score",
    "iou_score",
    "calculate_circularity",
] 