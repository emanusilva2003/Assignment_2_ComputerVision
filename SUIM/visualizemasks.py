import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Configuration
BASE_PATH = r"C:\Users\emanu\Desktop\Mestrado\2A\1S\VCOM\Assignment_2\SUIM\Pytorch\CKPT-PARA-ENVIAR"
TEST_IMAGES_PATH = r"C:\Users\emanu\Desktop\Mestrado\2A\1S\VCOM\Assignment_2\SUIM\TEST\images"
TEST_MASKS_PATH = r"C:\Users\emanu\Desktop\Mestrado\2A\1S\VCOM\Assignment_2\SUIM\TEST\masks"
MODELS = ['ckpt_rsb', 'ckpt_vgg', 'ckpt_swin', 'ckpt_rsb_aug', 'ckpt_vgg_aug', 'ckpt_swin_aug']
SELECTED_IMAGES = ['d_r_47_.bmp', 'f_r_644_.bmp', 'w_r_147_.bmp', 'f_r_1737_.bmp']

# Create 2x4 grid (adding original images and ground truth as first column)
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('RGB Masks Comparison - 4 Images per Model', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

# First, add original images (top-left) and ground truth masks (bottom-left)
# Original Images
original_images = []
target_size = (256, 320)  # Standard size for resizing
for img_name in SELECTED_IMAGES:
    # Convert .bmp to .jpg for original images
    jpg_name = img_name.replace('.bmp', '.jpg')
    img_path = os.path.join(TEST_IMAGES_PATH, jpg_name)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        original_images.append(np.array(img))
    else:
        print(f"Warning: Original image not found - {img_path}")

if len(original_images) > 0:
    # Pad to 4 images if needed
    while len(original_images) < 4:
        original_images.append(np.zeros_like(original_images[0]))
    
    # Create 2x2 grid for original images
    top_row = np.hstack([original_images[0], original_images[1]])
    bottom_row = np.hstack([original_images[2], original_images[3]])
    stacked_originals = np.vstack([top_row, bottom_row])
    
    axes[0].imshow(stacked_originals)
    axes[0].axis('off')
    axes[0].set_title('ORIGINAL IMAGES', fontsize=14, fontweight='bold', pad=10)
else:
    axes[0].text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=14)
    axes[0].axis('off')

# Ground Truth Masks
gt_masks = []
for img_name in SELECTED_IMAGES:
    gt_path = os.path.join(TEST_MASKS_PATH, img_name)
    if os.path.exists(gt_path):
        img = Image.open(gt_path)
        img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        gt_masks.append(np.array(img))
        found = True
    else:
        found = False

    if not found:
        print(f"Warning: Ground truth not found - {img_name}")

if len(gt_masks) > 0:
    # Pad to 4 images if needed
    while len(gt_masks) < 4:
        gt_masks.append(np.zeros_like(gt_masks[0]))
    
    # Create 2x2 grid for ground truth
    top_row = np.hstack([gt_masks[0], gt_masks[1]])
    bottom_row = np.hstack([gt_masks[2], gt_masks[3]])
    stacked_gt = np.vstack([top_row, bottom_row])
    
    axes[4].imshow(stacked_gt)
    axes[4].axis('off')
    axes[4].set_title('GROUND TRUTH', fontsize=14, fontweight='bold', pad=10)
else:
    axes[4].text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=14)
    axes[4].axis('off')

# Load and display stacked images for each model
model_positions = [1, 2, 3, 5, 6, 7]  # Skip positions 0 and 4 (used for originals and GT)
for idx, model in enumerate(MODELS):
    # Get RGB mask directory
    rgb_dir = os.path.join(BASE_PATH, model, 'Output', 'RGB')
    if os.path.exists(rgb_dir):
        # Load specific images
        images = []
        for img_file in SELECTED_IMAGES:
            img_path = os.path.join(rgb_dir, img_file)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
                images.append(np.array(img))
            else:
                print(f"Warning: Image not found - {img_path}")
                # Create blank placeholder
                if len(images) > 0:
                    images.append(np.zeros_like(images[0]))
                    images.append(np.zeros_like(images[0]))
        
        if len(images) > 0:
            # Pad to 4 images if needed
            while len(images) < 4:
                images.append(np.zeros_like(images[0]))
            
            # Create 2x2 grid
            top_row = np.hstack([images[0], images[1]])
            bottom_row = np.hstack([images[2], images[3]])
            stacked = np.vstack([top_row, bottom_row])
            
            # Display stacked image at correct position
            axes[model_positions[idx]].imshow(stacked)
            axes[model_positions[idx]].axis('off')
            
            # Set title
            model_name = model.replace('ckpt_', '').replace('_', ' ').upper()
            axes[model_positions[idx]].set_title(model_name, fontsize=14, fontweight='bold', pad=10)
        else:
            axes[model_positions[idx]].text(0.5, 0.5, 'No Images', ha='center', va='center', fontsize=14)
            axes[model_positions[idx]].axis('off')
    else:
        print(f"Warning: Directory not found - {rgb_dir}")
        axes[model_positions[idx]].text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=14)
        axes[model_positions[idx]].axis('off')

plt.tight_layout()
plt.savefig('masks_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Visualization saved as 'masks_comparison.png'")