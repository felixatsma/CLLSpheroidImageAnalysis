#!/usr/bin/env python3
"""
Usage:
    python demo.py [command] [options]
    
Commands:
    train       - Train a segmentation model
    analyze     - Analyze a single image
    extract     - Extract features from images
    demo        - Run all examples (default)
"""

import os
import sys
import argparse
import numpy as np
import torch  # Still needed for device detection
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cll_spheroid_analysis import (
    SpheroidAnalysisPipeline, 
    ImagePreprocessor, 
    ImageAugmenter,
    SpheroidSegmenter, 
    BlobDetector, 
    UNet, 
    DinoV2BinarySegmentation, 
    SegmentationDataset,
    SegmentationTrainer,
    create_dataloaders,
    train_segmentation_model,
    FeatureExtractor,
    show_mask, 
    show_mask_overlay, 
    plot_results,
    dice_score, 
    iou_score, 
    calculate_circularity
)

# Configuration
SAM2_CHECKPOINT = 'snellius/thesis/sam2/checkpoints/sam2.1_hiera_large.pt'
DEFAULT_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


# The train_model function is now part of the package as SegmentationTrainer


def example_train_model():
    """Example: Train a UNet model for blob segmentation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training a UNet Model for Blob Segmentation")
    print("="*60)
    
    # Check if training data exists
    data_dir = "data/augmented"  # Adjust path as needed
    if not os.path.exists(data_dir):
        print(f"Training data directory '{data_dir}' not found.")
        print("Please ensure you have training data in the correct format:")
        print("- Images: .jpg files")
        print("- Masks: corresponding '_masks.tif' files")
        print("Skipping training example...")
        return
    
    # Train the model using the package's training functionality
    try:
        trainer = train_segmentation_model(
            model_type='unet',
            image_dir=data_dir,
            resolution=(256, 256),  # Smaller resolution for faster training
            batch_size=4,
            epochs=5,  # Reduced for demo
            learning_rate=1e-4,
            device=DEFAULT_DEVICE,
            save_path='unet_demo_model.pth'
        )
        
        print("Training completed! Model saved as 'unet_demo_model.pth'")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("This might be due to missing dependencies or data.")


def example_analyze_image(image_path):
    """Example: Analyze a single image with visualization."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Image Analysis with Visualization")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        print("Please provide a valid image path.")
        return
    
    print(f"Analyzing image: {image_path}")
    
    # Initialize the pipeline
    pipeline = SpheroidAnalysisPipeline(
        spheroid_segmenter_config={
            'sam2_checkpoint': SAM2_CHECKPOINT,
            'device': DEFAULT_DEVICE
        },
        blob_detector_config={
            'method': 'traditional',  # Use traditional method for demo
            'device': DEFAULT_DEVICE
        }
    )
    
    # Run analysis
    try:
        result = pipeline.analyze_single_image(image_path, visualize=True)
        
        # Print extracted features
        print("\nExtracted features:")
        for feature, value in result['features'].items():
            print(f"  {feature}: {value:.4f}")
        
        # Show additional visualizations
        if 'spheroid_mask' in result and 'blob_mask' in result:
            print("\nShowing mask overlays...")
            
            # Load original image
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(image_array, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Spheroid mask
            ax2.imshow(image_array, cmap='gray')
            show_mask(result['spheroid_mask'], ax2, random_color=False)
            ax2.set_title('Spheroid Segmentation')
            ax2.axis('off')
            
            # Blob mask
            ax3.imshow(image_array, cmap='gray')
            show_mask(result['blob_mask'], ax3, random_color=True)
            ax3.set_title('Blob Detection')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("This might be due to missing SAM2 checkpoint or other dependencies.")


def example_extract_features(image_path):
    """Example: Extract features from an image."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Feature Extraction")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        return
    
    print(f"Extracting features from: {image_path}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Load and preprocess image
    preprocessor = ImagePreprocessor()
    image = preprocessor.load_image(image_path)
    
    # Extract features
    try:
        features = extractor.extract_all_features(image, np.ones_like(image))  # Use full image as spheroid mask for demo
        
        print("\nExtracted features:")
        print("-" * 40)
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                print(f"{feature_name:25s}: {value:10.4f}")
            else:
                print(f"{feature_name:25s}: {str(value)[:50]}...")
        
        # Show some feature statistics
        print("\nFeature Statistics:")
        print("-" * 40)
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        if numeric_features:
            values = list(numeric_features.values())
            print(f"Mean value: {np.mean(values):.4f}")
            print(f"Std deviation: {np.std(values):.4f}")
            print(f"Min value: {np.min(values):.4f}")
            print(f"Max value: {np.max(values):.4f}")
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")


def example_compare_methods(image_path):
    """Example: Compare different segmentation methods."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comparing Different Segmentation Methods")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        return
    
    print(f"Comparing methods on: {image_path}")
    
    # Load image
    preprocessor = ImagePreprocessor()
    image = preprocessor.load_image(image_path)
    
    # Initialize different blob detectors
    methods = {
        'Traditional': BlobDetector(method='traditional'),
        'DINOv2': BlobDetector(method='dinov2', device=DEFAULT_DEVICE),
        'UNet': BlobDetector(method='unet', device=DEFAULT_DEVICE)
    }
    
    results = {}
    
    # Run each method
    for method_name, detector in methods.items():
        try:
            print(f"Running {method_name} method...")
            blob_mask = detector.detect_blobs(image)
            results[method_name] = blob_mask
            
            # Calculate some basic metrics
            blob_count = np.sum(blob_mask > 0)
            total_area = np.sum(blob_mask)
            print(f"  - Detected {blob_count} blobs")
            print(f"  - Total blob area: {total_area} pixels")
            
        except Exception as e:
            print(f"  - {method_name} failed: {e}")
            results[method_name] = None
    
    # Visualize results
    if any(results.values()):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Show each method's result
        axes = [ax2, ax3, ax4]
        for idx, (method_name, mask) in enumerate(results.items()):
            if mask is not None and idx < len(axes):
                axes[idx].imshow(image, cmap='gray')
                show_mask(mask, axes[idx], random_color=True)
                axes[idx].set_title(f'{method_name} Detection')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


def run_all_examples(image_path):
    """Run all examples."""
    print("CLL Spheroid Analysis - Comprehensive Demo")
    print("="*60)
    print(f"Device: {DEFAULT_DEVICE}")
    print(f"SAM2 checkpoint: {SAM2_CHECKPOINT}")
    
    # Example 1: Training
    example_train_model()
    
    # Example 2: Analysis
    if image_path and os.path.exists(image_path):
        example_analyze_image(image_path)
    else:
        print("\nSkipping image analysis - no valid image path provided")
    
    # Example 3: Feature extraction
    if image_path and os.path.exists(image_path):
        example_extract_features(image_path)
    else:
        print("\nSkipping feature extraction - no valid image path provided")
    
    # Example 4: Method comparison
    if image_path and os.path.exists(image_path):
        example_compare_methods(image_path)
    else:
        print("\nSkipping method comparison - no valid image path provided")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


def main():
    global DEFAULT_DEVICE
    
    parser = argparse.ArgumentParser(description="CLL Spheroid Analysis - Comprehensive Demo")
    parser.add_argument('command', nargs='?', default='demo', 
                       choices=['train', 'analyze', 'extract', 'compare', 'demo'],
                       help='Command to run')
    parser.add_argument('image', nargs='?', type=str, 
                       help='Path to input image (required for analyze/extract/compare)')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                       help="Device for inference: 'cuda', 'cpu', or 'mps'")
    parser.add_argument('--data-dir', type=str, default='data/augmented',
                       help='Directory containing training data')
    
    args = parser.parse_args()
    
    # Update global device
    DEFAULT_DEVICE = args.device
    
    if args.command == 'train':
        example_train_model()
    elif args.command == 'analyze':
        if not args.image:
            print("Error: Image path required for 'analyze' command")
            sys.exit(1)
        example_analyze_image(args.image)
    elif args.command == 'extract':
        if not args.image:
            print("Error: Image path required for 'extract' command")
            sys.exit(1)
        example_extract_features(args.image)
    elif args.command == 'compare':
        if not args.image:
            print("Error: Image path required for 'compare' command")
            sys.exit(1)
        example_compare_methods(args.image)
    elif args.command == 'demo':
        run_all_examples(args.image)
    
    print("\nDone.")


if __name__ == "__main__":
    main() 