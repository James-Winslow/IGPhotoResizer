import os
import shutil
import random
from PIL import Image, ImageDraw
from random import randint

# Directories for frozen test set
test_input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_images"
frozen_test_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/frozen_test_images"

# Ensure directories exist
if not os.path.exists(test_input_dir):
    os.makedirs(test_input_dir)

if not os.path.exists(frozen_test_dir):
    os.makedirs(frozen_test_dir)

# Function to generate complex random images for the frozen test set
def generate_complex_images(num_images=50):
    random.seed(11) # set seed to 11 so that the random images are reproducible
    for i in range(num_images):
        # Random width and height between 500 and 1000 pixels
        width = randint(500, 1000)
        height = randint(500, 1000)

        # Create a random image with a solid color background
        img = Image.new('RGB', (width, height), color=(randint(0, 255), randint(0, 255), randint(0, 255)))

        # Add random shapes to the image
        draw = ImageDraw.Draw(img)
        for _ in range(10):  # Add 10 random shapes for more complexity
            shape_type = randint(0, 1)  # Randomly choose between circle and rectangle
            if shape_type == 0:
                # Random ellipses/circles
                x0, y0 = randint(0, width), randint(0, height)
                x1, y1 = x0 + randint(50, 150), y0 + randint(50, 150)
                draw.ellipse([x0, y0, x1, y1], fill=(randint(0, 255), randint(0, 255), randint(0, 255)))
            else:
                # Random rectangles
                x0, y0 = randint(0, width), randint(0, height)
                x1, y1 = x0 + randint(50, 150), y0 + randint(50, 150)
                draw.rectangle([x0, y0, x1, y1], fill=(randint(0, 255), randint(0, 255), randint(0, 255)))

        # Add random lines to the image for more complexity
        for _ in range(5):
            x0, y0 = randint(0, width), randint(0, height)
            x1, y1 = randint(0, width), randint(0, height)
            draw.line([x0, y0, x1, y1], fill=(randint(0, 255), randint(0, 255), randint(0, 255)), width=randint(1, 5))

        # Save the generated image to the test directory
        img_path = os.path.join(test_input_dir, f"complex_image_{i + 1}.jpg")  # Using 'complex_image' prefix
        img.save(img_path)
        print(f"Generated {img_path}")

# Function to copy only complex images to the frozen test set
def copy_to_frozen_test_set():
    for file_name in os.listdir(test_input_dir):
        # Only copy images with 'complex_image' prefix
        if 'complex_image' in file_name:
            test_img_path = os.path.join(test_input_dir, file_name)
            frozen_img_path = os.path.join(frozen_test_dir, file_name)
            shutil.copy(test_img_path, frozen_img_path)
            print(f"Copied {file_name} to frozen test set")

if __name__ == "__main__":
    generate_complex_images(num_images=50)  # Generate 20 images for the frozen set
    copy_to_frozen_test_set()
