import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(file_info):
    input_path, output_path = file_info

    with Image.open(input_path) as img:
        # Calculate new dimensions (half the original size)
        new_width = img.width // 2
        new_height = img.height // 2
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save the resized image to the output folder
        resized_img.save(output_path, "PNG")

def resize_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    file_info_list = []

    # Prepare file info for processing
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            file_info_list.append((input_path, output_path))

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:
        executor.map(process_image, file_info_list)

    print("All images resized and saved.")

# Example usage
input_folder = "C:/Users/Robert/Desktop/studia/Computer Vision/Project 3/data/GTAV/small/x"
output_folder = "C:/Users/Robert/Desktop/studia/Computer Vision/Project 3/data/GTAV/small/y"
resize_images(input_folder, output_folder)
