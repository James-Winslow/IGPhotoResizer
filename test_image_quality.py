import pandas as pd
from PIL import Image, ImageOps
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from resize import process_images  # Import your existing resizing function

# Define the directories
test_input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_images"
test_output_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_output"

# Ensure the test directories exist
if not os.path.exists(test_input_dir):
    os.makedirs(test_input_dir)

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Function to calculate SSIM and MSE between original and processed images
def calculate_metrics(original_img_path, processed_img_path):
    original_img = Image.open(original_img_path)
    processed_img = Image.open(processed_img_path)

    # Resize the processed image to match the dimensions of the original image
    processed_img_resized = processed_img.resize(original_img.size)

    # Convert to grayscale for SSIM
    original_gray = ImageOps.grayscale(original_img)
    processed_gray = ImageOps.grayscale(processed_img_resized)

    # Convert images to numpy arrays for SSIM and MSE calculation
    original_array = np.array(original_gray)
    processed_array = np.array(processed_gray)

    # Calculate SSIM
    ssim_value = ssim(original_array, processed_array)

    # Calculate MSE
    mse_value = np.mean((original_array - processed_array) ** 2)

    return ssim_value, mse_value

# Function to evaluate and record results for a single run
def evaluate_images(n=1):
    results = []
    
    # Run the resizing and evaluation n times
    for i in range(n):
        # Run the resizing process
        process_images(test_input_dir, test_output_dir)
        
        # Evaluate each image in the folder
        for file_name in os.listdir(test_input_dir):
            original_img_path = os.path.join(test_input_dir, file_name)
            processed_img_path = os.path.join(test_output_dir, file_name)

            if os.path.exists(processed_img_path):
                ssim_value, mse_value = calculate_metrics(original_img_path, processed_img_path)
                results.append({
                    'Image': file_name,
                    'SSIM': ssim_value,
                    'MSE': mse_value
                })

        # Aggregate the results into a DataFrame
        results_df = pd.DataFrame(results)
        print(f"Results for Run {i+1}:")
        print(results_df)
        
        # Save results to a CSV for later analysis
        results_df.to_csv(f'results_run_{i+1}.csv', index=False)

    return results_df

# Main function to run the tests
if __name__ == "__main__":
    # Test with n=1 first
    evaluate_images(n=1)
