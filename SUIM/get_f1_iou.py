"""
# Script for evaluating F score and mIOU for all 5 categories
# Saves results to Excel file with format matching the paper
"""

from __future__ import print_function, division
import ntpath
import numpy as np
from PIL import Image
import pandas as pd
from os.path import exists
# local libs
from utils.data_utils import getPaths
from utils.measure_utils import db_eval_boundary, IoU_bin

## Configuration
model_name = "SUIM-Net_VGG_Aug"  # Change this for different models
test_dir = "SUIM/TEST/masks/"
gen_base_dir = "SUIM/TEST/Pytorch_output_VGG_Aug/"
output_excel = "SUIM/TEST/evaluation_results.xlsx"

## Categories to evaluate
categories = ["HD", "WR", "RO", "RI", "FV"]

## input/output shapes
im_res = (320, 256) # Ajustar apenas para os modelos originais

# for reading and scaling input images
def read_and_bin(im_path):
    img = Image.open(im_path).resize(im_res)
    img = np.array(img)/255.
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def evaluate_category(category):
    """Evaluate F-score and IoU for a single category"""
    real_mask_dir = test_dir + category + "/"
    gen_mask_dir = gen_base_dir + category + "/"
    
    if not exists(real_mask_dir) or not exists(gen_mask_dir):
        print(f"Warning: Directory not found for {category}")
        return None, None, None, None
    
    Ps, Rs, F1s, IoUs = [], [], [], []
    gen_paths = sorted(getPaths(gen_mask_dir))
    real_paths = sorted(getPaths(real_mask_dir))
    
    if len(gen_paths) == 0 or len(real_paths) == 0:
        print(f"Warning: No images found for {category}")
        return None, None, None, None
    
    for gen_p, real_p in zip(gen_paths, real_paths):
        gen, real = read_and_bin(gen_p), read_and_bin(real_p)
        if (np.sum(real)>0):
            precision, recall, F1 = db_eval_boundary(real, gen)
            iou = IoU_bin(real, gen)
            Ps.append(precision) 
            Rs.append(recall)
            F1s.append(F1)
            IoUs.append(iou)
    
    if len(F1s) == 0:
        return None, None, None, None
    
    # Calculate mean and std
    f_mean = 100.0 * np.mean(F1s)
    f_std = 100.0 * np.std(F1s)
    iou_mean = 100.0 * np.mean(IoUs)
    iou_std = 100.0 * np.std(IoUs)
    
    return f_mean, f_std, iou_mean, iou_std

# Evaluate all categories
print(f"Evaluating model: {model_name}")
print("=" * 60)

results_f = {}
results_iou = {}
valid_f_values = []
valid_iou_values = []

for category in categories:
    print(f"Processing {category}...", end=" ")
    f_mean, f_std, iou_mean, iou_std = evaluate_category(category)
    
    if f_mean is not None:
        results_f[category] = f"{f_mean:.2f} ± {f_std:.2f}"
        results_iou[category] = f"{iou_mean:.2f} ± {iou_std:.2f}"
        valid_f_values.append(f_mean)
        valid_iou_values.append(iou_mean)
        print(f"F1: {f_mean:.2f}±{f_std:.2f}, IoU: {iou_mean:.2f}±{iou_std:.2f}")
    else:
        results_f[category] = "N/A"
        results_iou[category] = "N/A"
        print("Failed")

# Calculate Combined (average across all categories)
if valid_f_values:
    combined_f_mean = np.mean(valid_f_values)
    combined_f_std = np.std(valid_f_values)
    combined_iou_mean = np.mean(valid_iou_values)
    combined_iou_std = np.std(valid_iou_values)
    
    results_f["Combined"] = f"{combined_f_mean:.2f} ± {combined_f_std:.2f}"
    results_iou["Combined"] = f"{combined_iou_mean:.2f} ± {combined_iou_std:.2f}"
    
    print("\n" + "=" * 60)
    print(f"Combined - F1: {combined_f_mean:.2f}±{combined_f_std:.2f}, IoU: {combined_iou_mean:.2f}±{combined_iou_std:.2f}")

# Create DataFrames
df_f = pd.DataFrame([results_f], index=[model_name])
df_iou = pd.DataFrame([results_iou], index=[model_name])

# Load existing Excel or create new one
if exists(output_excel):
    with pd.ExcelFile(output_excel) as xls:
        if 'F-Score' in xls.sheet_names:
            existing_f = pd.read_excel(xls, 'F-Score', index_col=0)
            df_f = pd.concat([existing_f, df_f])
            df_f = df_f[~df_f.index.duplicated(keep='last')]
        
        if 'mIoU' in xls.sheet_names:
            existing_iou = pd.read_excel(xls, 'mIoU', index_col=0)
            df_iou = pd.concat([existing_iou, df_iou])
            df_iou = df_iou[~df_iou.index.duplicated(keep='last')]

# Save to Excel with two sheets
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df_f.to_excel(writer, sheet_name='F-Score')
    df_iou.to_excel(writer, sheet_name='mIoU')

print("\n" + "=" * 60)
print(f"Results saved to {output_excel}")
print(f"Sheet 'F-Score': F1 scores for all categories")
print(f"Sheet 'mIoU': IoU scores for all categories")
