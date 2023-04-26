import os

import numpy as np
from PIL import Image

# Define the paths to the directories containing the image files and mosaic
image_dir = '/home/mrchen/cmr/mosaic/KAIR/sr3/datasets/english_easy_train/hr_(128,512)'
mosaic_dir = '/home/mrchen/cmr/mosaic/KAIR/sr3/datasets/english_easy_train/lr_(32,128)'
output_dir = '/home/mrchen/cmr/mosaic/DiG/npy_dir'
# Initialize the dictionaries to store the image file paths and labels
gt_info_dict = {}
mosaic_gt_info_dict = {}

# Loop over all the image files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        # Load the image file into a numpy array
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        img_arr = np.array(img)
        
        # Load the corresponding mosaic file into a numpy array
        mosaic_path = os.path.join(mosaic_dir, filename)
        mosaic = Image.open(mosaic_path)
        mosaic_arr = np.array(mosaic)
        
        # Add the image and mosaic file paths and labels to the dictionaries
        gt_info_dict[img_path] = img_arr
        mosaic_gt_info_dict[mosaic_path] = mosaic_arr
        
# Save the dictionaries as .npy files
np.save(os.path.join(output_dir, 'gt.npy'), gt_info_dict)
np.save(os.path.join(output_dir, 'mosaic_info.npy'), mosaic_gt_info_dict)
