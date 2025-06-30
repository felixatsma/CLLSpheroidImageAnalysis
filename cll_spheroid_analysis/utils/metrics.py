"""
Evaluation metrics for spheroid analysis.
"""

import numpy as np
from typing import Union, Tuple
import cv2


def dice_score(pred_mask, true_mask, smooth=1e-6):
    """
    Calculate Dice coefficient between predicted and true masks.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient score
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)
    return float(dice)


def iou_score(pred_mask, true_mask, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) between predicted and true masks.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def calculate_circularity(contour):
    """
    Calculate circularity of a contour.
    
    Args:
        contour: Contour points array
        
    Returns:
        Circularity score (1.0 = perfect circle)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return float(circularity)


def calculate_area(mask, pixel_size=1.0):
    """
    Calculate area of a binary mask.
    
    Args:
        mask: Binary mask
        pixel_size: Size of each pixel in micrometers
        
    Returns:
        Area in square micrometers
    """
    area_pixels = np.sum(mask > 0)
    return float(area_pixels * (pixel_size ** 2))


def calculate_perimeter(mask, pixel_size=1.0):
    """
    Calculate perimeter of a binary mask.
    
    Args:
        mask: Binary mask
        pixel_size: Size of each pixel in micrometers
        
    Returns:
        Perimeter in micrometers
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    # Sum up all contour perimeters
    total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return float(total_perimeter * pixel_size)


def calculate_radius(mask, pixel_size=1.0):
    """
    Calculate equivalent radius of a binary mask.
    
    Args:
        mask: Binary mask
        pixel_size: Size of each pixel in micrometers
        
    Returns:
        Equivalent radius in micrometers
    """
    area = calculate_area(mask, pixel_size)
    return float(np.sqrt(area / np.pi))


def calculate_compactness(mask):
    """
    Calculate compactness (isoperimetric ratio) of a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Compactness score (1.0 = perfect circle)
    """
    area = calculate_area(mask)
    perimeter = calculate_perimeter(mask)
    
    if perimeter == 0:
        return 0.0
    
    compactness = (4 * np.pi * area) / (perimeter * perimeter)
    return float(compactness)


def calculate_eccentricity(mask):
    """
    Calculate eccentricity of a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Eccentricity score (0.0 = perfect circle, 1.0 = line)
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 5:
        return 0.0
    
    # Fit ellipse
    try:
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(largest_contour)
        
        if minor_axis == 0:
            return 1.0
        
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return float(eccentricity)
    except:
        return 0.0


def calculate_all_metrics(pred_mask, true_mask):
    """
    Calculate all evaluation metrics for segmentation results.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'dice': dice_score(pred_mask, true_mask),
        'iou': iou_score(pred_mask, true_mask),
        'area_pred': calculate_area(pred_mask),
        'area_true': calculate_area(true_mask),
        'perimeter_pred': calculate_perimeter(pred_mask),
        'perimeter_true': calculate_perimeter(true_mask),
        'radius_pred': calculate_radius(pred_mask),
        'radius_true': calculate_radius(true_mask),
        'circularity_pred': calculate_compactness(pred_mask),
        'circularity_true': calculate_compactness(true_mask),
        'eccentricity_pred': calculate_eccentricity(pred_mask),
        'eccentricity_true': calculate_eccentricity(true_mask),
    }
    
    return metrics 