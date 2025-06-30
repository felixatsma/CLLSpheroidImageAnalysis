"""
Image augmentation utilities for spheroid analysis.
"""

import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple, Union
import random


class ImageAugmenter:
    """
    Class for augmenting spheroid images and their corresponding masks.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the image augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def augment_image(self, image, mask=None):
        """
        Perform simple augmentations: flips and rotations.
        Returns a list of augmented images with their augmentation names.
        
        Args:
            image: Input image
            mask: Corresponding mask (optional)
            
        Returns:
            List of tuples (augmentation_name, augmented_image)
        """
        augmentations = []
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray(mask)
            else:
                mask_pil = mask
        
        # Original
        augmentations.append(('original', image_pil))
        
        # Horizontal flip
        aug_img = ImageOps.mirror(image_pil)
        augmentations.append(('flip_horizontal', aug_img))
        
        # Vertical flip
        aug_img = ImageOps.flip(image_pil)
        augmentations.append(('flip_vertical', aug_img))
        
        # Rotations
        for angle in [90, 180, 270]:
            aug_img = image_pil.rotate(angle)
            augmentations.append((f'rotate_{angle}', aug_img))
        
        # Convert back to numpy arrays if input was numpy
        if isinstance(image, np.ndarray):
            augmentations = [(name, np.array(img)) for name, img in augmentations]
        
        return augmentations
    
    def augment_with_mask(self, image, mask):
        """
        Augment both image and mask together.
        
        Args:
            image: Input image
            mask: Corresponding mask
            
        Returns:
            List of tuples (augmentation_name, augmented_image, augmented_mask)
        """
        augmentations = []
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask)
        else:
            mask_pil = mask
        
        # Original
        augmentations.append(('original', image_pil, mask_pil))
        
        # Horizontal flip
        aug_img = ImageOps.mirror(image_pil)
        aug_mask = ImageOps.mirror(mask_pil)
        augmentations.append(('flip_horizontal', aug_img, aug_mask))
        
        # Vertical flip
        aug_img = ImageOps.flip(image_pil)
        aug_mask = ImageOps.flip(mask_pil)
        augmentations.append(('flip_vertical', aug_img, aug_mask))
        
        # Rotations
        for angle in [90, 180, 270]:
            aug_img = image_pil.rotate(angle)
            aug_mask = mask_pil.rotate(angle)
            augmentations.append((f'rotate_{angle}', aug_img, aug_mask))
        
        # Convert back to numpy arrays if input was numpy
        if isinstance(image, np.ndarray):
            augmentations = [(name, np.array(img), np.array(mask)) for name, img, mask in augmentations]
        
        return augmentations
    
    def random_augment(self, image, mask=None, num_augmentations=3):
        """
        Apply random augmentations to an image.
        
        Args:
            image: Input image
            mask: Corresponding mask (optional)
            num_augmentations: Number of random augmentations to apply
            
        Returns:
            List of augmented images
        """
        all_augmentations = self.augment_image(image, mask)
        
        # Remove original from random selection
        random_augs = [aug for aug in all_augmentations if aug[0] != 'original']
        
        # Select random augmentations
        selected = random.sample(random_augs, min(num_augmentations, len(random_augs)))
        
        return [aug[1] for aug in selected]
    
    def save_augmentations(self, image, output_dir, base_name, mask=None):
        """
        Save all augmentations to files.
        
        Args:
            image: Input image
            output_dir: Directory to save augmented images
            base_name: Base name for the files
            mask: Corresponding mask (optional)
            
        Returns:
            List of saved file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        if mask is not None:
            augmentations = self.augment_with_mask(image, mask)
            
            for aug_name, aug_img, aug_mask in augmentations:
                # Save image
                img_path = os.path.join(output_dir, f"{base_name}_{aug_name}.jpg")
                if isinstance(aug_img, np.ndarray):
                    Image.fromarray(aug_img).save(img_path)
                else:
                    aug_img.save(img_path)
                saved_paths.append(img_path)
                
                # Save mask
                mask_path = os.path.join(output_dir, f"{base_name}_{aug_name}_mask.tif")
                if isinstance(aug_mask, np.ndarray):
                    Image.fromarray(aug_mask).save(mask_path)
                else:
                    aug_mask.save(mask_path)
                saved_paths.append(mask_path)
        else:
            augmentations = self.augment_image(image)
            
            for aug_name, aug_img in augmentations:
                img_path = os.path.join(output_dir, f"{base_name}_{aug_name}.jpg")
                if isinstance(aug_img, np.ndarray):
                    Image.fromarray(aug_img).save(img_path)
                else:
                    aug_img.save(img_path)
                saved_paths.append(img_path)
        
        return saved_paths 