# Instagram Photo Resizer & Quality Optimizer
This project aims to optimize image resizing techniques for Instagram multi-photo posts, ensuring that images maintain their original quality as much as possible when resized.

## Project Overview
Instagram often resizes images for multi-photo posts based on the first image selected, which can lead to unwanted scaling and quality loss. This project explores various resizing methods to retain the image quality while adapting to the specific format constraints of Instagram. The project includes the following methods:

Simple Resize: Resizes images directly to a fixed dimension using LANCZOS resampling.
Padding Resize: Adds padding around the image to create uniform dimensions while maintaining the aspect ratio.
Content-Aware Resize: A more advanced method that resizes the image with minimal distortion, maintaining the content quality.
## Methods and Results
Methods
We tested three main resizing methods:

Simple Resize: Direct resizing of images to a fixed size.
Padding Resize: Adding padding to maintain the aspect ratio while ensuring images fit within the Instagram constraints.
Content-Aware Resize: Uses content-aware resizing techniques to avoid distortion while scaling.
Results
The results were evaluated using the following metrics:

SSIM (Structural Similarity Index): Measures how similar the resized image is to the original.
MSE (Mean Squared Error): Measures the error between the original and resized images.
Results for both test-generated images and real-world images are included in the CSV file resizing_comparison_results_real_images.csv. Note: The actual real-world images are not included in the repository for privacy reasons.

You can find the detailed results and comparison in the CSV file located in the results folder.

## Usage
Installation
To use this project, you need the following dependencies:

Python 3.x
PIL (Pillow)
OpenCV
Numpy
Scikit-Image
Pandas

## Running the Code
Running the Project
To resize images using the different methods, follow these steps:

Place your images in a folder (for example, frozen_real_images/).
Update the input_dir and output_dir paths in the script to reflect the folder locations.
To run the resizing methods and save the results:
python main.py

The resized images will be saved in the specified output directory, and the results will be saved in the resizing_comparison_results_real_images.csv.

Real-World Image Testing
Real-world images were tested as part of the project to assess how the resizing methods perform on common use-case images. The results from this testing are included in the results CSV file but the images themselves are not provided in this repository.

Future Enhancements
More Resizing Techniques: Explore additional resizing algorithms, such as neural network-based upscaling.
Batch Processing Interface: Develop a simple UI for batch image processing.
Instagram API Integration: Directly post resized images to Instagram.
Contribution
Feel free to fork this repository and contribute! Please submit pull requests with detailed descriptions of the changes made.