from PIL import Image, ImageDraw
import random
import os
from resize import process_images


# Define the directories
test_input_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_images"
test_output_dir = "/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/test_output"

# Ensure the test directories exist
if not os.path.exists(test_input_dir):
    os.makedirs(test_input_dir)

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Function to generate random images with solid backgrounds and shapes
def generate_random_images(num_images=10, min_size=300, max_size=1080):
    for i in range(num_images):
        # Random width and height between min and max sizes
        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)

        # Create a random image with a solid color background
        background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (width, height), background_color)

        # Add random shapes to the image
        draw = ImageDraw.Draw(img)
        for _ in range(10):  # Add 10 random shapes
            shape_type = random.choice(['circle', 'rectangle', 'polygon'])
            if shape_type == 'circle':
                x0, y0 = random.randint(0, width//2), random.randint(0, height//2)
                x1, y1 = x0 + random.randint(20, 100), y0 + random.randint(20, 100)
                draw.ellipse([x0, y0, x1, y1], fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
            elif shape_type == 'rectangle':
                x0, y0 = random.randint(0, width//2), random.randint(0, height//2)
                x1, y1 = x0 + random.randint(20, 100), y0 + random.randint(20, 100)
                draw.rectangle([x0, y0, x1, y1], fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
            else:  # Polygon
                points = [(random.randint(0, width), random.randint(0, height)) for _ in range(6)]
                draw.polygon(points, fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

        # Save the image to the test directory
        img_path = os.path.join(test_input_dir, f"random_image_{i+1}.jpg")
        img.save(img_path)
        print(f"Generated {img_path}")

# Main function to run the tests
def run_image_tests():
    # Step 1: Generate random images
    generate_random_images()

    # Step 2: Apply the resizing algorithm to the random images
    process_images(test_input_dir, test_output_dir)
    print(f"Test completed. Resized images saved in {test_output_dir}")

if __name__ == "__main__":
    run_image_tests()
