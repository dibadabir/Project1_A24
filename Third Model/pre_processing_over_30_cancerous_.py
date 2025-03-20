# -*- coding: utf-8 -*-
"""pre processing over 30 cancerous .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17S2Xev9MR8YPINzBqhIFYv8bAKgAp4v2
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_hair_direct_replace(
    image,
    blackhat_kernel_size=(9, 9),
    blackhat_threshold=10,
    inpaint_radius=1
):
    """
    Removes hair from the image by:
      1. Detecting hair regions using blackhat + threshold + morphological ops.
      2. Inpainting only those regions.
      3. Directly replacing hair pixels in the original image with inpainted pixels,
         preserving the original color in non-hair areas.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blackhat to highlight hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, blackhat_kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a binary mask
    _, hair_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening (remove small noise)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, open_kernel)

    # Dilate to ensure thin hair strands are fully covered
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # Inpaint the image in hair regions
    inpainted = cv2.inpaint(image, hair_mask, inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Convert hair_mask to boolean for indexing
    hair_mask_bool = hair_mask.astype(bool)

    # Create a copy of the original image
    result = image.copy()

    # Directly replace hair pixels in 'result' with the inpainted pixels
    result[hair_mask_bool] = inpainted[hair_mask_bool]

    return result

def process_and_save_images(input_folder, output_folder):
    """
    Processes all images in the input folder by removing hair via direct replacement,
    resizing them to 224x224, and saving the results in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error reading image: {image_file}. Skipping...")
            continue

        # Remove hair with direct replacement approach
        processed_image = remove_hair_direct_replace(image)

        # Resize to 224x224
        processed_image = cv2.resize(processed_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)

# Example usage (adjust paths as needed)
input_folder = "/content/drive/MyDrive/ Discipline-specific/Organised Images/mel"
output_folder = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 mel"

process_and_save_images(input_folder, output_folder)
print("Image preprocessing complete!")

"""pre processing for over 30 bcc

"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_hair_direct_replace(
    image,
    blackhat_kernel_size=(9, 9),
    blackhat_threshold=10,
    inpaint_radius=1
):
    """
    Removes hair from the image by:
      1. Detecting hair regions using blackhat + threshold + morphological ops.
      2. Inpainting only those regions.
      3. Directly replacing hair pixels in the original image with inpainted pixels,
         preserving the original color in non-hair areas.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blackhat to highlight hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, blackhat_kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a binary mask
    _, hair_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening (remove small noise)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, open_kernel)

    # Dilate to ensure thin hair strands are fully covered
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # Inpaint the image in hair regions
    inpainted = cv2.inpaint(image, hair_mask, inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Convert hair_mask to boolean for indexing
    hair_mask_bool = hair_mask.astype(bool)

    # Create a copy of the original image
    result = image.copy()

    # Directly replace hair pixels in 'result' with the inpainted pixels
    result[hair_mask_bool] = inpainted[hair_mask_bool]

    return result

def process_and_save_images(input_folder, output_folder):
    """
    Processes all images in the input folder by removing hair via direct replacement,
    resizing them to 224x224, and saving the results in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error reading image: {image_file}. Skipping...")
            continue

        # Remove hair with direct replacement approach
        processed_image = remove_hair_direct_replace(image)

        # Resize to 224x224
        processed_image = cv2.resize(processed_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)

# Example usage (adjust paths as needed)
input_folder = "/content/drive/MyDrive/ Discipline-specific/Organised Images/bcc"
output_folder = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 bcc"

process_and_save_images(input_folder, output_folder)
print("Image preprocessing complete!")

"""pre processing for over 30 akiec"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_hair_direct_replace(
    image,
    blackhat_kernel_size=(9, 9),
    blackhat_threshold=10,
    inpaint_radius=1
):
    """
    Removes hair from the image by:
      1. Detecting hair regions using blackhat + threshold + morphological ops.
      2. Inpainting only those regions.
      3. Directly replacing hair pixels in the original image with inpainted pixels,
         preserving the original color in non-hair areas.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blackhat to highlight hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, blackhat_kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a binary mask
    _, hair_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening (remove small noise)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, open_kernel)

    # Dilate to ensure thin hair strands are fully covered
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # Inpaint the image in hair regions
    inpainted = cv2.inpaint(image, hair_mask, inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Convert hair_mask to boolean for indexing
    hair_mask_bool = hair_mask.astype(bool)

    # Create a copy of the original image
    result = image.copy()

    # Directly replace hair pixels in 'result' with the inpainted pixels
    result[hair_mask_bool] = inpainted[hair_mask_bool]

    return result

def process_and_save_images(input_folder, output_folder):
    """
    Processes all images in the input folder by removing hair via direct replacement,
    resizing them to 224x224, and saving the results in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error reading image: {image_file}. Skipping...")
            continue

        # Remove hair with direct replacement approach
        processed_image = remove_hair_direct_replace(image)

        # Resize to 224x224
        processed_image = cv2.resize(processed_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)

# Example usage (adjust paths as needed)
input_folder = "/content/drive/MyDrive/ Discipline-specific/Organised Images/akiec"
output_folder = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 ekiec"

process_and_save_images(input_folder, output_folder)
print("Image preprocessing complete!")

"""data augmentaion for akiec"""

import os
import math
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Input folder containing multiple images
input_dir = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 ekiec"

# Output folder for augmented images
output_folder = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 akiec with data aumentation"
os.makedirs(output_folder, exist_ok=True)

# List all valid image files in the input folder
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
num_images = len(image_files)
if num_images == 0:
    raise ValueError("No image files found in the directory.")

print(f"Found {num_images} images in the input folder.")

# Set target total augmented images
target_augmented = 600

# Calculate number of augmentations per image (rounded up)
aug_per_image = math.ceil(target_augmented / num_images)
print(f"Each image will produce up to {aug_per_image} augmented images.")

# Create an ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.9, 1.1),
    fill_mode='reflect'
)

generated_total = 0

print("Starting data augmentation...")

# Loop over all images in the input folder
for image_file in tqdm(image_files, desc="Augmenting images"):
    input_path = os.path.join(input_dir, image_file)

    # Load the image and convert it to an array
    img = load_img(input_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # shape: (1, height, width, channels)

    # Create a generator for this image
    aug_generator = datagen.flow(
        x,
        batch_size=1,
        save_to_dir=output_folder,
        save_prefix='aug',
        save_format='jpg'
    )

    # Generate augmented images for this file
    for _ in range(aug_per_image):
        if generated_total >= target_augmented:
            break  # Stop if we've reached the target
        next(aug_generator)  # Generates and saves 1 new image
        generated_total += 1

    if generated_total >= target_augmented:
        break  # Break out of the outer loop if target reached

print(f"Data augmentation complete! Total augmented images created: {generated_total}")