"""
Visualization utilities for spheroid analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import cv2


def show_mask(mask, ax, random_color=False, borders=True):
    """
    Display a mask overlay on a matplotlib axis.
    
    Args:
        mask: Binary mask array
        ax: Matplotlib axis to plot on
        random_color: Whether to use random color or default blue
        borders: Whether to draw contour borders
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)


def show_mask_overlay(image, spheroid_mask=None, blob_mask=None, figsize=(10, 8)):
    """
    Display an image with spheroid and blob mask overlays.
    
    Args:
        image: Input image array
        spheroid_mask: Binary mask of the spheroid
        blob_mask: Binary mask of detected blobs (optional)
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    
    # Show spheroid mask in blue
    if spheroid_mask is not None:
        spheroid_binary = (spheroid_mask > 0).astype(np.uint8)
        show_mask(spheroid_binary, plt.gca(), random_color=False, borders=True)
    
    # Show blob mask in red if provided
    if blob_mask is not None:
        blob_binary = (blob_mask > 0).astype(np.uint8)
        # Create a red color for blobs
        h, w = blob_binary.shape
        blob_color = np.array([255/255, 0/255, 0/255, 0.6])  # Red with alpha
        blob_mask_image = blob_binary.reshape(h, w, 1) * blob_color.reshape(1, 1, -1)
        plt.imshow(blob_mask_image)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_image_mask(image, mask, figsize=(16, 8)):
    """
    Display original image and mask side by side.
    
    Args:
        image: Input image array
        mask: Binary mask array
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title('Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_results(original_image, spheroid_mask, blob_mask=None, 
                true_spheroid_mask=None, true_blob_mask=None, figsize=(15, 10)):
    """
    Plot comprehensive results including original image, predicted masks, and optionally true masks.
    
    Args:
        original_image: Original input image
        spheroid_mask: Predicted spheroid segmentation mask
        blob_mask: Predicted blob segmentation mask (optional)
        true_spheroid_mask: Ground truth spheroid mask (optional)
        true_blob_mask: Ground truth blob mask (optional)
        figsize: Figure size for the plot
    """
    # Determine number of subplots
    num_plots = 2  # Original + spheroid
    if blob_mask is not None:
        num_plots += 1
    if true_spheroid_mask is not None:
        num_plots += 1
    if true_blob_mask is not None:
        num_plots += 1
    
    plt.figure(figsize=figsize)
    
    # Original image
    plt.subplot(2, (num_plots + 1) // 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Predicted spheroid mask
    plt.subplot(2, (num_plots + 1) // 2, 2)
    plt.imshow(spheroid_mask, cmap='gray')
    plt.title('Predicted Spheroid Mask')
    plt.axis('off')
    
    plot_idx = 3
    
    # Predicted blob mask
    if blob_mask is not None:
        plt.subplot(2, (num_plots + 1) // 2, plot_idx)
        plt.imshow(blob_mask, cmap='gray')
        plt.title('Predicted Blob Mask')
        plt.axis('off')
        plot_idx += 1
    
    # True spheroid mask
    if true_spheroid_mask is not None:
        plt.subplot(2, (num_plots + 1) // 2, plot_idx)
        plt.imshow(true_spheroid_mask, cmap='gray')
        plt.title('True Spheroid Mask')
        plt.axis('off')
        plot_idx += 1
    
    # True blob mask
    if true_blob_mask is not None:
        plt.subplot(2, (num_plots + 1) // 2, plot_idx)
        plt.imshow(true_blob_mask, cmap='gray')
        plt.title('True Blob Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_masks(image, mask, borders=True, figsize=(10, 10)):
    """
    Display image with mask overlay using the show_mask function.
    
    Args:
        image: Input image array
        mask: Binary mask array
        borders: Whether to draw contour borders
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    show_mask(mask, plt.gca(), borders=borders)
    plt.axis('off')
    plt.show() 