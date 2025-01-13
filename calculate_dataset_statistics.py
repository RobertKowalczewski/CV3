import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def calculate_mean_std(image_path):
    """Calculate the mean and standard deviation for a single image."""
    try:
        with Image.open(image_path) as img:
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            mean = img_array.mean(axis=(0, 1))
            std = img_array.std(axis=(0, 1))
            return mean, std
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_images_in_folder(folder_path):
    """Iterate through all PNG images in the folder and calculate mean/std for each."""
    png_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    means = []
    stds = []

    def process_file(file_path):
        mean, std = calculate_mean_std(file_path)
        if mean is not None and std is not None:
            means.append(mean)
            stds.append(std)

    with ThreadPoolExecutor() as executor:
        executor.map(process_file, png_files)

    return means, stds

def main():
    folder_path = input("Enter the path to the folder containing PNG images: ").strip()
    if not os.path.exists(folder_path):
        print("Invalid folder path.")
        return

    means, stds = process_images_in_folder(folder_path)

    print(f"Mean: {np.mean(means, axis=0)}\nStd: {np.mean(stds, axis=0)}\n")

if __name__ == "__main__":
    main()
