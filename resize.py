from PIL import Image, ImageOps
from colorthief import ColorThief
import os

# Define the directories
input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/Images"
output_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/output"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to get the dominant color from the image
def get_dominant_color(img_path):
    color_thief = ColorThief(img_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

# Proportional resizing with padding
def resize_with_padding(img, target_width, target_height, dominant_color):
    width, height = img.size
    aspect_ratio = width / height

    # Proportional scaling
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with target dimensions and background color
    new_img = Image.new("RGB", (target_width, target_height), dominant_color)
    
    # Center the resized image within the new image (with padding)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    new_img.paste(img_resized, (x_offset, y_offset))

    return new_img

# Main function to process all images in the input directory
def process_images(input_dir, output_dir):
    # Ensure the output directory is cleared
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the target width and height
    target_width = 1080  # e.g., Instagram's width limit
    target_height = 1350  # e.g., Instagram’s 4:5 aspect ratio height

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            
            # Get the dominant color from the image
            dominant_color = get_dominant_color(img_path)
            
            # Resize the image with padding
            img_resized = resize_with_padding(img, target_width, target_height, dominant_color)
            
            # Save the processed image
            output_path = os.path.join(output_dir, file_name)
            img_resized.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    process_images(input_dir, output_dir)
