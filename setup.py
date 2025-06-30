from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cll-spheroid-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for analyzing CLL spheroid images using SAM2 for spheroid segmentation and multiple methods for blob detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CLLSpheroidImageAnalysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.20.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "segment-anything-2",
    ],
    include_package_data=True,
    package_data={
        "cll_spheroid_analysis": ["models/*.pth", "configs/*.yaml"],
    },
) 