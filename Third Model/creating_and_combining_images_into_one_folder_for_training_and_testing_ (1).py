# -*- coding: utf-8 -*-
"""Creating and combining images into one folder for training and testing .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ASReo_J4JfF9DSxBxyDeOh9A-Fgc_A5K
"""

import os
import shutil
import random

def combine_images_from_two_folders():
    # Hardcoded folder paths
    folder1 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 mel"  # Folder for 500 images
    folder2 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /under 30 mel"  # Folder for 200 images
    destination_folder = "/content/drive/MyDrive/Discipline-specific/Final images folder for cancerous model /mel"  # Destination folder

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List all valid image files in each folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images1 = [f for f in os.listdir(folder1) if f.lower().endswith(valid_extensions)]
    images2 = [f for f in os.listdir(folder2) if f.lower().endswith(valid_extensions)]

    # Shuffle the lists so we pick random images
    random.shuffle(images1)
    random.shuffle(images2)

    # Pick the first 500 images from folder1 (or all if fewer than 500 exist)
    picked1 = images1[:500]

    # Pick the first 200 images from folder2 (or all if fewer than 200 exist)
    picked2 = images2[:200]

    # Copy and rename the images to the destination folder
    count_copied = 0

    for img_file in picked1:
        src_path = os.path.join(folder1, img_file)
        dst_path = os.path.join(destination_folder, "mel_" + img_file)
        shutil.copy(src_path, dst_path)
        count_copied += 1

    for img_file in picked2:
        src_path = os.path.join(folder2, img_file)
        dst_path = os.path.join(destination_folder, "mel_" + img_file)
        shutil.copy(src_path, dst_path)
        count_copied += 1

    print(f"Done! Copied {count_copied} images in total to {destination_folder}.")

# Run the function
combine_images_from_two_folders()

"""akiec"""

import os
import shutil
import random

def combine_images_from_two_folders():
    # Hardcoded folder paths
    folder1 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 akiec with data aumentation"  # Folder for 500 images
    folder2 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /under 30 akiec"  # Folder for 200 images
    destination_folder = "/content/drive/MyDrive/Discipline-specific/Final images folder for cancerous model /akiec"  # Destination folder

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List all valid image files in each folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images1 = [f for f in os.listdir(folder1) if f.lower().endswith(valid_extensions)]
    images2 = [f for f in os.listdir(folder2) if f.lower().endswith(valid_extensions)]

    # Debug prints: how many images are in each folder?
    print(f"Folder1 has {len(images1)} valid images.")
    print(f"Folder2 has {len(images2)} valid images.")

    # Shuffle the lists so we pick random images
    random.shuffle(images1)
    random.shuffle(images2)

    # Pick the first 500 images from folder1 (or all if fewer than 500 exist)
    picked1 = images1[:500]
    # Pick the first 200 images from folder2 (or all if fewer than 200 exist)
    picked2 = images2[:200]

    count_copied = 0
    unique_counter = 1

    for img_file in picked1:
        src_path = os.path.join(folder1, img_file)
        new_name = f"akiec_{unique_counter}_{img_file}"
        dst_path = os.path.join(destination_folder, new_name)
        shutil.copy(src_path, dst_path)
        count_copied += 1
        unique_counter += 1

    for img_file in picked2:
        src_path = os.path.join(folder2, img_file)
        new_name = f"akiec_{unique_counter}_{img_file}"
        dst_path = os.path.join(destination_folder, new_name)
        shutil.copy(src_path, dst_path)
        count_copied += 1
        unique_counter += 1

    print(f"Done! Copied {count_copied} images in total to {destination_folder}.")

# Run the function
combine_images_from_two_folders()

"""bcc

"""

import os
import shutil
import random

def combine_images_from_two_folders():
    # Hardcoded folder paths
    folder1 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /over 30 bcc"  # Folder for 500 images
    folder2 = "/content/drive/MyDrive/Discipline-specific/Pre-processed and data augmentated images /under 30 bcc"  # Folder for 200 images
    destination_folder = "/content/drive/MyDrive/Discipline-specific/Final images folder for cancerous model /bcc"  # Destination folder

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List all valid image files in each folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images1 = [f for f in os.listdir(folder1) if f.lower().endswith(valid_extensions)]
    images2 = [f for f in os.listdir(folder2) if f.lower().endswith(valid_extensions)]

    # Debug prints: how many images are in each folder?
    print(f"Folder1 has {len(images1)} valid images.")
    print(f"Folder2 has {len(images2)} valid images.")

    # Shuffle the lists so we pick random images
    random.shuffle(images1)
    random.shuffle(images2)

    # Pick the first 500 images from folder1 (or all if fewer than 500 exist)
    picked1 = images1[:500]
    # Pick the first 200 images from folder2 (or all if fewer than 200 exist)
    picked2 = images2[:200]

    count_copied = 0
    unique_counter = 1

    for img_file in picked1:
        src_path = os.path.join(folder1, img_file)
        new_name = f"bcc_{unique_counter}_{img_file}"
        dst_path = os.path.join(destination_folder, new_name)
        shutil.copy(src_path, dst_path)
        count_copied += 1
        unique_counter += 1

    for img_file in picked2:
        src_path = os.path.join(folder2, img_file)
        new_name = f"bcc_{unique_counter}_{img_file}"
        dst_path = os.path.join(destination_folder, new_name)
        shutil.copy(src_path, dst_path)
        count_copied += 1
        unique_counter += 1

    print(f"Done! Copied {count_copied} images in total to {destination_folder}.")

# Run the function
combine_images_from_two_folders()