"""
Training utilities for segmentation models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import os

from .models import UNet, DinoV2BinarySegmentation
from .dataset import SegmentationDataset


class SegmentationTrainer:
    """
    Trainer class for segmentation models.
    """
    
    def __init__(self, 
                 model,
                 device='auto',
                 learning_rate=1e-4,
                 weight_decay=1e-5):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to train on ('auto', 'cuda', 'mps', 'cpu')
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for optimization
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader,
              val_loader,
              epochs=10,
              save_best=True,
              save_path='best_model.pth',
              early_stopping_patience=None):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_best: Whether to save the best model
            save_path: Path to save the best model
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary containing training history
        """
        print(f"Training on {self.device} for {epochs} epochs...")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(save_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_model(self, path: str):
        """Save the model state dict."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load the model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def create_dataloaders(image_dir,
                      mask_dir=None,
                      resolution=(256, 256),
                      batch_size=4,
                      train_split=0.8,
                      num_workers=2,
                      augment=True):
    """
    Create training and validation dataloaders.
    
    Args:
        image_dir: Directory containing image files
        mask_dir: Directory containing mask files (if None, assumes masks are in image_dir)
        resolution: Target resolution for images (height, width)
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        resolution=resolution
    )
    
    # Split into train/validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def train_segmentation_model(model_type='unet',
                           image_dir='data/augmented',
                           mask_dir=None,
                           resolution=(256, 256),
                           batch_size=4,
                           epochs=10,
                           learning_rate=1e-4,
                           device='auto',
                           save_path='best_model.pth',
                           **kwargs):
    """
    Convenience function to train a segmentation model.
    
    Args:
        model_type: Type of model ('unet' or 'dinov2')
        image_dir: Directory containing training images
        mask_dir: Directory containing training masks
        resolution: Target resolution for images
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save the trained model
        **kwargs: Additional arguments for the trainer
        
    Returns:
        Trained SegmentationTrainer instance
    """
    # Create model
    if model_type.lower() == 'unet':
        model = UNet(n_channels=1, n_classes=1)
    elif model_type.lower() == 'dinov2':
        model = DinoV2BinarySegmentation()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        image_dir=image_dir,
        mask_dir=mask_dir,
        resolution=resolution,
        batch_size=batch_size
    )
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path,
        **kwargs
    )
    
    return trainer 