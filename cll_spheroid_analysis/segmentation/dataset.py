"""
Dataset classes for spheroid segmentation.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SegmentationDataset(Dataset):
    """
    Dataset class for spheroid segmentation with images and masks.
    """
    
    def __init__(self, image_dir, mask_dir=None, 
                 resolution=(960, 1280), transform=None):
        """
        Initialize the segmentation dataset.
        
        Args:
            image_dir: Directory containing image files
            mask_dir: Directory containing mask files (if None, assumes masks are in image_dir with '_masks.tif' suffix)
            resolution: Target resolution for images (height, width)
            transform: Optional custom transforms
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.resolution = resolution
        
        # Get list of image files
        self.images = [img for img in os.listdir(image_dir) 
                      if os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        
        # Basic transforms if none provided
        self.default_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image-mask pair.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Determine mask path
        if self.mask_dir is not None:
            mask_name = img_name.replace('.jpg', '_masks.tif').replace('.png', '_masks.tif')
            mask_path = os.path.join(self.mask_dir, mask_name)
        else:
            mask_name = img_name.replace('.jpg', '_masks.tif').replace('.png', '_masks.tif')
            mask_path = os.path.join(self.image_dir, mask_name)
        
        # Load image and mask
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            mask = Image.open(mask_path).convert('L')  # Grayscale for binary mask
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            # Return dummy data if loading fails
            image = Image.new('L', self.resolution, 0)
            mask = Image.new('L', self.resolution, 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)

        mask = (self.mask_transform(mask) > 0).float()
        
        return image, mask
    
    def get_image_paths(self):
        """
        Get list of all image paths.
        
        Returns:
            List of image file paths
        """
        return [os.path.join(self.image_dir, img) for img in self.images]
    
    def get_mask_paths(self):
        """
        Get list of all mask paths.
        
        Returns:
            List of mask file paths
        """
        mask_paths = []
        for img_name in self.images:
            if self.mask_dir is not None:
                mask_name = img_name.replace('.jpg', '_masks.tif').replace('.png', '_masks.tif')
                mask_path = os.path.join(self.mask_dir, mask_name)
            else:
                mask_name = img_name.replace('.jpg', '_masks.tif').replace('.png', '_masks.tif')
                mask_path = os.path.join(self.image_dir, mask_name)
            mask_paths.append(mask_path)
        return mask_paths


class AugmentedSegmentationDataset(Dataset):
    """
    Dataset class that includes data augmentation.
    """
    
    def __init__(self, image_dir, mask_dir=None,
                 resolution=(960, 1280), augment=True):
        """
        Initialize the augmented segmentation dataset.
        
        Args:
            image_dir: Directory containing image files
            mask_dir: Directory containing mask files
            resolution: Target resolution for images
            augment: Whether to apply augmentations
        """
        self.base_dataset = SegmentationDataset(image_dir, mask_dir, resolution)
        self.augment = augment
        
        if augment:
            self.augmentation_transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.augmentation_transform = self.base_dataset.default_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get a single image-mask pair with optional augmentation.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        image, mask = self.base_dataset[idx]
        
        if self.augment:
            # Apply augmentation to image
            image = self.augmentation_transform(image)
            # Apply same transforms to mask (without random augmentations)
            mask = self.base_dataset.mask_transform(mask)
            mask = (mask > 0).float()
        
        return image, mask 