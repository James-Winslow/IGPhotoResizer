import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from random import randint

# Directories for frozen test set
test_input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_images"
frozen_test_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/frozen_test_images"

# Ensure directories exist
if not os.path.exists(test_input_dir):
    os.makedirs(test_input_dir)

if not os.path.exists(frozen_test_dir):
    os.makedirs(frozen_test_dir)

# Function to generate random images for the frozen test set
def generate_test_images(num_images=10):
    for i in range(num_images):
        img = Image.new('RGB', (randint(500, 1000), randint(500, 1000)), color=(randint(0, 255), randint(0, 255), randint(0, 255)))
        img_path = os.path.join(test_input_dir, f"random_image_{i+1}.jpg")
        img.save(img_path)
        print(f"Generated {img_path}")

# Function to copy to frozen test set
def copy_to_frozen_test_set():
    for file_name in os.listdir(test_input_dir):
        test_img_path = os.path.join(test_input_dir, file_name)
        frozen_img_path = os.path.join(frozen_test_dir, file_name)
        shutil.copy(test_img_path, frozen_img_path)
        print(f"Copied {file_name} to frozen test set")

if __name__ == "__main__":
    generate_test_images(num_images=20)  # Generate 20 images for the frozen set
    copy_to_frozen_test_set()
