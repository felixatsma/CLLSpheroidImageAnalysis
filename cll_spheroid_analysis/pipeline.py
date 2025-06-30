"""
Main pipeline for CLL spheroid analysis.
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd

from .preprocessing import ImagePreprocessor
from .segmentation import SpheroidSegmenter, BlobDetector
from .feature_extraction import FeatureExtractor
from .utils.visualization import show_mask_overlay, plot_results
from .utils.metrics import dice_score, iou_score


class SpheroidAnalysisPipeline:
    """
    Complete pipeline for analyzing CLL spheroid images.
    """
    
    def __init__(self, 
                 preprocessor_config=None,
                 spheroid_segmenter_config=None,
                 blob_detector_config=None,
                 feature_config=None):
        """
        Initialize the analysis pipeline.
        
        Args:
            preprocessor_config: Configuration for image preprocessing
            spheroid_segmenter_config: Configuration for spheroid segmentation (SAM2)
            blob_detector_config: Configuration for blob detection
            feature_config: Configuration for feature extraction
        """
        # Default configurations
        self.preprocessor_config = preprocessor_config or {
            'target_size': (1280, 960),
            'normalize': True,
            'convert_to_grayscale': False  # Keep RGB for SAM2
        }
        
        self.spheroid_segmenter_config = spheroid_segmenter_config or {
            'sam2_checkpoint': None,
            'device': None
        }
        
        self.blob_detector_config = blob_detector_config or {
            'method': 'traditional',  # 'traditional', 'unet', 'dinov2'
            'model_path': None,
            'device': None
        }
        
        # Store threshold separately (not passed to BlobDetector constructor)
        self.blob_detection_threshold = blob_detector_config.get('threshold', 150) if blob_detector_config else 150
        
        self.feature_config = feature_config or {
            'pixel_size': 1.0
        }
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(**self.preprocessor_config)
        self.spheroid_segmenter = SpheroidSegmenter(**self.spheroid_segmenter_config)
        self.blob_detector = BlobDetector(**self.blob_detector_config)
        self.feature_extractor = FeatureExtractor(**self.feature_config)
    
    def analyze_single_image(self, image_path, 
                           true_spheroid_mask=None,
                           true_blob_mask=None,
                           visualize=False):
        """
        Analyze a single spheroid image.
        
        Args:
            image_path: Path to the image file
            true_spheroid_mask: Ground truth spheroid mask (optional, for evaluation)
            true_blob_mask: Ground truth blob mask (optional, for evaluation)
            visualize: Whether to show visualization plots
            
        Returns:
            Dictionary containing analysis results
        """
        # Preprocess image for SAM2 (RGB)
        image_rgb = self.preprocessor.preprocess(image_path, mean_std_normalize=False)
        
        # Segment spheroid using SAM2 (needs RGB)
        spheroid_mask = self.spheroid_segmenter.segment_spheroid(image_rgb)
        
        # Convert to grayscale for blob detection
        if len(image_rgb.shape) == 3:
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_rgb
        
        # Detect blobs using selected method
        blob_mask = self.blob_detector.detect_blobs(
            image_gray, 
            spheroid_mask, 
            threshold=self.blob_detection_threshold
        )
        
        # Extract features
        features = self.feature_extractor.extract_all_features(image_gray, spheroid_mask, blob_mask)
        
        # Prepare results
        results = {
            'image_path': str(image_path),
            'spheroid_mask': spheroid_mask,
            'blob_mask': blob_mask,
            'features': features
        }
        
        # Evaluate if ground truth is provided
        if true_spheroid_mask is not None or true_blob_mask is not None:
            evaluation_metrics = self._evaluate_segmentation(
                spheroid_mask, blob_mask, true_spheroid_mask, true_blob_mask
            )
            results['evaluation'] = evaluation_metrics
        
        # Visualize if requested
        if visualize:
            if true_spheroid_mask is not None or true_blob_mask is not None:
                plot_results(image_gray, spheroid_mask, blob_mask, true_spheroid_mask, true_blob_mask)
            else:
                show_mask_overlay(image_gray, spheroid_mask, blob_mask)
        
        return results
    
    def analyze_batch(self, image_paths, 
                     true_masks=None,
                     visualize=False):
        """
        Analyze a batch of spheroid images.
        
        Args:
            image_paths: List of image file paths
            true_masks: List of ground truth masks (optional)
            visualize: Whether to show visualization plots
            
        Returns:
            List of analysis result dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            true_mask = true_masks[i] if true_masks is not None else None
            result = self.analyze_single_image(image_path, true_mask, visualize)
            results.append(result)
        
        return results
    
    def analyze_directory(self, image_dir, 
                         mask_dir=None,
                         output_file=None):
        """
        Analyze all images in a directory and save results to CSV.
        
        Args:
            image_dir: Directory containing image files
            mask_dir: Directory containing ground truth masks (optional)
            output_file: Path to save results CSV (optional)
            
        Returns:
            DataFrame containing all analysis results
        """
        image_dir = Path(image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.tif"))
        
        true_masks = None
        if mask_dir is not None:
            mask_dir = Path(mask_dir)
            true_masks = []
            for img_path in image_paths:
                mask_name = img_path.stem + "_masks.tif"
                mask_path = mask_dir / mask_name
                if mask_path.exists():
                    from PIL import Image
                    mask = np.array(Image.open(mask_path))
                    true_masks.append(mask)
                else:
                    true_masks.append(None)
        
        # Analyze all images
        results = self.analyze_batch(image_paths, true_masks)
        
        # Convert to DataFrame
        df = self._results_to_dataframe(results)
        
        # Save to file if specified
        if output_file is not None:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return df
    
    def _evaluate_segmentation(self, pred_spheroid_mask, pred_blob_mask, true_spheroid_mask, true_blob_mask):
        """
        Evaluate segmentation performance.
        
        Args:
            pred_spheroid_mask: Predicted spheroid mask
            pred_blob_mask: Predicted blob mask
            true_spheroid_mask: Ground truth spheroid mask
            true_blob_mask: Ground truth blob mask
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'dice_score_spheroid': dice_score(pred_spheroid_mask, true_spheroid_mask),
            'iou_score_spheroid': iou_score(pred_spheroid_mask, true_spheroid_mask),
            'dice_score_blob': dice_score(pred_blob_mask, true_blob_mask),
            'iou_score_blob': iou_score(pred_blob_mask, true_blob_mask)
        }
        return metrics
    
    def _results_to_dataframe(self, results):
        """
        Convert results list to pandas DataFrame.
        
        Args:
            results: List of analysis result dictionaries
            
        Returns:
            DataFrame with all results
        """
        # Extract features and metadata
        data = []
        for result in results:
            row = {
                'image_path': result['image_path'],
                **result['features']
            }
            
            # Add evaluation metrics if available
            if 'evaluation' in result:
                row.update(result['evaluation'])
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_masks(self, results, output_dir):
        """
        Save segmentation masks to files.
        
        Args:
            results: List of analysis results
            output_dir: Directory to save masks
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            image_path = Path(result['image_path'])
            mask = result['spheroid_mask']
            
            # Save mask with same name as image but with _mask suffix
            mask_name = f"{image_path.stem}_mask.tif"
            mask_path = output_dir / mask_name
            
            from PIL import Image
            Image.fromarray(mask * 255).save(mask_path)
    
    def get_summary_statistics(self, results):
        """
        Calculate summary statistics from analysis results.
        
        Args:
            results: List of analysis results
            
        Returns:
            Dictionary of summary statistics
        """
        df = self._results_to_dataframe(results)
        
        # Calculate statistics for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        
        return summary.to_dict()
    
    def compare_with_ground_truth(self, results):
        """
        Compare predictions with ground truth and provide detailed metrics.
        
        Args:
            results: List of analysis results with evaluation metrics
            
        Returns:
            Dictionary of comparison metrics
        """
        # Filter results that have evaluation metrics
        evaluated_results = [r for r in results if 'evaluation' in r]
        
        if not evaluated_results:
            return {}
        
        # Calculate average metrics
        dice_scores = [r['evaluation']['dice_score_spheroid'] for r in evaluated_results]
        iou_scores = [r['evaluation']['iou_score_spheroid'] for r in evaluated_results]
        
        comparison = {
            'num_evaluated': len(evaluated_results),
            'mean_dice_score_spheroid': np.mean(dice_scores),
            'std_dice_score_spheroid': np.std(dice_scores),
            'mean_iou_score_spheroid': np.mean(iou_scores),
            'std_iou_score_spheroid': np.std(iou_scores),
            'min_dice_score_spheroid': np.min(dice_scores),
            'max_dice_score_spheroid': np.max(dice_scores),
            'min_iou_score_spheroid': np.min(iou_scores),
            'max_iou_score_spheroid': np.max(iou_scores)
        }
        
        return comparison 