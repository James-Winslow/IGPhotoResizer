# IGPhotoResizer

## Overview
This project provides a robust tool for resizing and padding a set of images for optimal posting on Instagram using the multi-photo feature. The tool ensures that vertical and horizontal images with different aspect ratios maintain their original orientation without automatic scaling issues from Instagram.

It now includes functionality to evaluate different resizing strategies using a frozen test set of images, with metrics such as SSIM (Structural Similarity Index Measure) and MSE (Mean Squared Error) to assess image quality.

## How It Works
Image Resizing: The tool resizes photos to meet Instagram's aspect ratio requirements (4:5) while preserving important parts of the image.
Padding: Adds padding based on the image's dominant colors, maintaining aesthetic coherence and avoiding black bars.
Frozen Test Set: A frozen set of randomly generated images is used to consistently test different resizing strategies, ensuring reliable performance evaluation.
Quality Metrics: The resizing methods are evaluated based on SSIM and MSE to ensure minimal loss of quality during resizing.

## How to Use
1. Run create_frozen_test_set.py to generate and freeze a set of images for testing.
2. Run test_image_quality.py to evaluate image resizing strategies and generate quality metrics over multiple runs.
3. Upload the processed images to Instagram via the multi-photo post feature.

## Running the Code
'#' Navigate to the project directory
cd IGPhotoResizer

'#' Install the required dependencies
pip install -r requirements.txt

'#' Run the scripts as needed
python create_frozen_test_set.py  # Create a frozen set of test images
python test_image_quality.py      # Run the resizing and evaluation process



## Future Plans
Machine Learning Integration: We plan to integrate machine learning to optimize the resizing strategy based on the test results.
Multiple Resizing Methods: Testing and comparing multiple resizing algorithms to determine the best quality-preserving method.
