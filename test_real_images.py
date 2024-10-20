import pandas as pd
from PIL import Image
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Unified results DataFrame
results = pd.DataFrame(columns=['File', 'Method', 'SSIM', 'MSE'])

# Calculate metrics function
def calculate_metrics(img1, img2):
    # Ensure both images are grayscale and calculate SSIM and MSE
    img1_gray = np.array(img1.convert('L'))
    img2_gray = np.array(img2.convert('L'))

    # SSIM requires same dimensions, so crop larger image to match smaller
    min_height = min(img1_gray.shape[0], img2_gray.shape[0])
    min_width = min(img1_gray.shape[1], img2_gray.shape[1])
    
    img1_cropped = img1_gray[:min_height, :min_width]
    img2_cropped = img2_gray[:min_height, :min_width]

    # Calculate SSIM and MSE
    ssim_value = ssim(img1_cropped, img2_cropped)
    mse_value = np.mean((img1_cropped - img2_cropped) ** 2)

    return ssim_value, mse_value

# Resizing functions that append results to a single DataFrame
def process_images(input_dir, output_dir, method_name, resize_function):
    ensure_output_dir(output_dir)
    global results  # Using global results DataFrame to collect all results
    all_results = []  # Accumulate results in a list before concatenating

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img_original = Image.open(img_path)
            img_resized = resize_function(img_original)  # Call method-specific resize function

            # Calculate metrics by comparing the original image to the resized image
            ssim_value, mse_value = calculate_metrics(img_original, img_resized)

            # Append to the results list
            all_results.append({
                'File': file_name,
                'Method': method_name,
                'SSIM': ssim_value,
                'MSE': mse_value
            })

            # Save resized image (optional)
            output_path = os.path.join(output_dir, file_name)
            img_resized.save(output_path)

            print(f"{method_name} resize completed for: {file_name}")

    # Concatenate the results list into the global results DataFrame
    results = pd.concat([results, pd.DataFrame(all_results)], ignore_index=True)

# Example resize functions
def simple_resize(img):
    # Resize by scaling down width and keeping aspect ratio
    scale_factor = 0.5
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def padding_resize(img):
    # Resize with padding to maintain aspect ratio
    max_size = max(img.size)
    new_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))  # White background
    new_img.paste(img, (int((max_size - img.size[0]) / 2), int((max_size - img.size[1]) / 2)))
    return new_img.resize((800, 800), Image.Resampling.LANCZOS)  # Resizing to fixed size for comparison

def content_aware_resize(img):
    # Content-aware resizing using OpenCV
    img_np = np.array(img)
    target_width = int(img_np.shape[1] * 0.5)  # Scale down width by 50%
    new_size = (target_width, int(img_np.shape[0] * (target_width / img_np.shape[1])))
    img_resized = cv2.resize(img_np, new_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_resized)

# Ensure output directory
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Main function to run all methods
def main():
    input_dir = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/frozen_real_images'
    output_dir = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/resized_images'

    # Process with all resizing methods
    process_images(input_dir, output_dir, 'Simple Resize', simple_resize)
    process_images(input_dir, output_dir, 'Padding Resize', padding_resize)
    process_images(input_dir, output_dir, 'Content-Aware Resize', content_aware_resize)

    # Save the results to CSV
    results_file = os.path.join(output_dir, 'resizing_comparison_results_real_images.csv')
    results.to_csv(results_file, index=False)
    print(f"All results saved to {results_file}")

if __name__ == "__main__":
    main()
