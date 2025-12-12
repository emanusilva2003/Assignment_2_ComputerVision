"""
# Script to create RGB combined masks from existing binary masks
# Reads the individual binary masks (RO, FV, HD, RI, WR) and combines them into RGB visualizations
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
# local libs
from utils.data_utils import getPaths, binaryMasksToRGB

## Directories for binary masks
output_dir = "SUIM/TEST/output/"
RO_dir = output_dir + "RO/"
FV_dir = output_dir + "FV/"
WR_dir = output_dir + "WR/"
HD_dir = output_dir + "HD/"
RI_dir = output_dir + "RI/"
RGB_dir = output_dir + "RGB/"

# Create RGB output directory if it doesn't exist
if not exists(RGB_dir): 
    os.makedirs(RGB_dir)

# Get all mask files from one of the directories (they should all have the same files)
mask_paths = getPaths(RO_dir)

print(f"Found {len(mask_paths)} masks to process")

for ro_path in mask_paths:
    # Get the filename
    img_name = ntpath.basename(ro_path)
    
    # Build paths for all 5 masks
    fv_path = join(FV_dir, img_name)
    hd_path = join(HD_dir, img_name)
    ri_path = join(RI_dir, img_name)
    wr_path = join(WR_dir, img_name)
    
    # Check if all masks exist
    if not all([exists(ro_path), exists(fv_path), exists(hd_path), exists(ri_path), exists(wr_path)]):
        print(f"Skipping {img_name} - not all masks found")
        continue
    
    # Read all binary masks
    RO = np.array(Image.open(ro_path)) / 255.0
    FV = np.array(Image.open(fv_path)) / 255.0
    HD = np.array(Image.open(hd_path)) / 255.0
    RI = np.array(Image.open(ri_path)) / 255.0
    WR = np.array(Image.open(wr_path)) / 255.0
    
    # Create RGB combined mask
    rgb_mask = binaryMasksToRGB(RO, FV, HD, RI, WR)
    
    # Save RGB mask
    Image.fromarray(rgb_mask).save(join(RGB_dir, img_name))
    print(f"Created RGB mask for: {img_name}")

print(f"\nDone! RGB masks saved in: {RGB_dir}")
