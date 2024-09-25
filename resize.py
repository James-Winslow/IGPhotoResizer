from PIL import Image
from colorthief import ColorThief
import os

# Define the directories
input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/Images"
output_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/output"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to clear the output directory
def clear_output_dir(output_dir):
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Function to get the dominant color from the image
def get_dominant_color(img_path):
    color_thief = ColorThief(img_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

# Function to find the largest image in the directory
def find_largest_image(input_dir):
    max_width = 0
    max_height = 0
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            width, height = img.size
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
    return max_width, max_height

# Strategy for smaller images: Add padding without resizing
def handle_small_image(img, target_width, target_height, dominant_color):
    width, height = img.size
    # Create a new image with the target dimensions and the dominant color
    new_img = Image.new("RGB", (target_width, target_height), dominant_color)
    # Paste the original image in the center without resizing
    new_img.paste(img, ((target_width - width) // 2, (target_height - height) // 2))
    return new_img

# Strategy for larger images: Proportional scaling with minimal padding
def handle_large_image(img, target_width, target_height, dominant_color):
    width, height = img.size
    aspect_ratio = width / height

    # Proportional scaling
    if width > height:
        scale_factor = target_width / width
    else:
        scale_factor = target_height / height

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target dimensions and the dominant color
    new_img = Image.new("RGB", (target_width, target_height), dominant_color)
    # Paste the resized image in the center
    new_img.paste(img_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_img

# Main function to process all images in the input directory
def process_images(input_dir, output_dir):
    # Clear the output directory
    clear_output_dir(output_dir)

    # Find the dimensions of the largest image
    max_width, max_height = find_largest_image(input_dir)

    # Set a reasonable target width and height based on the max image size
    target_width = min(max_width, 1080)  # Limit the maximum width (e.g., Instagram’s 1080px width)
    target_height = int(target_width / (4 / 5))  # Use 4:5 aspect ratio

    # Define a threshold for "small" images (e.g., images with width or height below 720px)
    size_threshold = 360

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            width, height = img.size

            # Get the dominant color of the image
            dominant_color = get_dominant_color(img_path)

            # Apply the two-prong strategy
            if width < size_threshold or height < size_threshold:
                # For smaller images, do not scale, just add padding
                img_with_padding = handle_small_image(img, target_width, target_height, dominant_color)
            else:
                # For larger images, scale proportionally and add minimal padding
                img_with_padding = handle_large_image(img, target_width, target_height, dominant_color)

            # Save the processed image
            output_path = os.path.join(output_dir, file_name)
            img_with_padding.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    process_images(input_dir, output_dir)
