# CLL Spheroid Image Analysis

A Python package for analyzing images of spheroids of leukemia cells. This package provides a pipeline for preprocessing, segmenting spheroids using SAM2, detecting blobs using traditional image processing or neural networks, extracting features, and in the future, performing predictions on spheroid images.

This package was made for a thesis project 'Development of a CLL Spheroid Image Analysis Pipeline'.

## Installation

```bash
git clone https://github.com/yourusername/CLLSpheroidImageAnalysis.git
cd CLLSpheroidImageAnalysis

pip install -e .
```
Install SAM2 from https://github.com/facebookresearch/segment-anything-2


### Usage

See demo.py for example usage of the package.

## Components

The `ImagePreprocessor` class handles image loading, resizing, normalization, and enhancement.
The `SpheroidSegmenter` class uses SAM2 for spheroid masking.
The `BlobDetector` class provides multiple methods for CLL cluster (blob) detection.
The `FeatureExtractor` class extracts various features from spheroids and blobs.
