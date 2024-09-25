# IGPhotoResizer

## Overview
This project provides a simple tool to resize and pad a set of images for optimal posting on Instagram using the multi-photo feature. It ensures that vertical and horizontal images with different aspect ratios maintain their original orientation without automatic scaling issues from Instagram. 

## How It Works
- **Image Resizing**: The tool resizes photos to meet Instagram's aspect ratio requirements (4:5) without losing important parts of the image.
- **Padding**: Adds black padding to images where necessary, preserving the image quality and dimensions.

## How to Use
1. Place the photos you want to resize in the `images/` folder.
2. Run the Python script to process the images and save the results in the `output/` folder.
3. Upload the processed images to Instagram via the multi-photo post feature.

## Running the Code
```bash
# Navigate to the project directory
cd IGPhotoResizer

# Install the required dependencies
pip install -r requirements.txt

# Run the script
python src/resize_photos.py
