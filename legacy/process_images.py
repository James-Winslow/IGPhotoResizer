from PIL import Image
import os
import cv2  # Required for content-aware and feature-based resizing

# Ensure the output directory exists
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Simple Resize method
def process_images_simple(input_dir, output_dir):
    ensure_output_dir(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            img_resized = img.resize((800, 800), Image.Resampling.LANCZOS)  # Updated from Image.ANTIALIAS
            output_path = os.path.join(output_dir, file_name)
            img_resized.save(output_path)
            print(f"Simple resize completed for: {file_name}")

# Padding Resize method
def process_images_padding(input_dir, output_dir):
    ensure_output_dir(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            max_size = max(img.size)
            new_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))  # White background
            new_img.paste(img, (int((max_size - img.size[0]) / 2), int((max_size - img.size[1]) / 2)))
            output_path = os.path.join(output_dir, file_name)
            new_img.save(output_path)
            print(f"Padding resize completed for: {file_name}")

# Content-Aware Resize (Feature-Based) method using OpenCV
def process_images_content_aware(input_dir, output_dir):
    ensure_output_dir(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path)

            # Resize to a target size, maintaining aspect ratio (example 800px width)
            target_width = 800
            height, width = img.shape[:2]
            scale = target_width / width
            new_size = (target_width, int(height * scale))
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, img_resized)
            print(f"Content-aware resize completed for: {file_name}")

# Main function to run the resizing on different methods
def main(test_set='frozen_real_images'):
    input_dir = f'/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/{test_set}'
    
    # Output directories for different methods
    output_dir_simple = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/output/simple_resize'
    output_dir_padding = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/output/padding_resize'
    output_dir_content_aware = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/output/content_aware_resize'

    # Running all methods
    print("Starting Simple Resize...")
    process_images_simple(input_dir, output_dir_simple)
    
    print("Starting Padding Resize...")
    process_images_padding(input_dir, output_dir_padding)
    
    print("Starting Content-Aware Resize...")
    process_images_content_aware(input_dir, output_dir_content_aware)

if __name__ == '__main__':
    main()
