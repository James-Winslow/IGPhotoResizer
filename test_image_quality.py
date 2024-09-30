import time
import pandas as pd
from PIL import Image, ImageOps
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from resize import process_images  # Import your existing resizing function

# Directories
frozen_test_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/frozen_test_images"
test_output_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_output"

# Ensure directories exist
if not os.path.exists(frozen_test_dir):
    os.makedirs(frozen_test_dir)

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Function to calculate SSIM and MSE
def calculate_metrics(original_img_path, processed_img_path):
    original_img = Image.open(original_img_path)
    processed_img = Image.open(processed_img_path)

    # Resize processed image to match original dimensions
    processed_img_resized = processed_img.resize(original_img.size)

    # Convert to grayscale for SSIM
    original_gray = ImageOps.grayscale(original_img)
    processed_gray = ImageOps.grayscale(processed_img_resized)

    original_array = np.array(original_gray)
    processed_array = np.array(processed_gray)

    # Calculate SSIM and MSE
    ssim_value = ssim(original_array, processed_array)
    mse_value = np.mean((original_array - processed_array) ** 2)

    return ssim_value, mse_value

# Function to evaluate and aggregate results across multiple runs
def evaluate_images(n_runs=50, set_size=10):
    results = []
    
    for i in range(n_runs):
        print(f"Starting run {i + 1}...")
        start_time = time.time()  # Start timing

        # Run the resizing process
        process_images(frozen_test_dir, test_output_dir)

        # Evaluate each image in the frozen test folder
        for file_name in os.listdir(frozen_test_dir):
            original_img_path = os.path.join(frozen_test_dir, file_name)
            processed_img_path = os.path.join(test_output_dir, file_name)

            if os.path.exists(processed_img_path):
                ssim_value, mse_value = calculate_metrics(original_img_path, processed_img_path)
                results.append({
                    'Run': i + 1,
                    'Image': file_name,
                    'SSIM': ssim_value,
                    'MSE': mse_value
                })

        end_time = time.time()  # End timing
        runtime = end_time - start_time
        print(f"Run {i + 1} completed in {runtime:.2f} seconds")

    # Aggregate results into a DataFrame
    results_df = pd.DataFrame(results)
    print("Aggregated Results for Multiple Runs:")
    print(results_df)

    # Save results to a CSV file
    results_df.to_csv('aggregated_results.csv', index=False)

    return results_df

if __name__ == "__main__":
    # Run the experiment with 50 iterations of 10 images each
    evaluate_images(n_runs=5, set_size=10)
