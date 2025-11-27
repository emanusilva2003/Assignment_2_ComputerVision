"""
# Test script for SUIM-Net PyTorch
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists, dirname, abspath
import time
import torch
from torch.utils.data import DataLoader

# Local imports
import sys
script_dir = dirname(abspath(__file__))
parent_dir = dirname(script_dir)  # Go up one level to SUIM/Pytorch/
sys.path.append(parent_dir)

# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import SUIMDataset, binaryMasksToRGB

## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "SUIM/TEST/images/"

## sample and ckpt dir
samples_dir = "SUIM/TEST/Pytorch_output/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/"
RGB_dir = samples_dir + "RGB/"

if not exists(samples_dir): os.makedirs(samples_dir)
if not exists(RO_dir): os.makedirs(RO_dir)
if not exists(FB_dir): os.makedirs(FB_dir)
if not exists(WR_dir): os.makedirs(WR_dir)
if not exists(HD_dir): os.makedirs(HD_dir)
if not exists(RI_dir): os.makedirs(RI_dir)
if not exists(RGB_dir): os.makedirs(RGB_dir)

## Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## input/output shapes
base_ = 'RSB' # 'VGG' or 'RSB'
if base_=='RSB':
    im_res_ = (320, 256, 3) 
    ckpt_name = "suimnet_rsb.pth"
else: 
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.pth"

print("\n" + "="*60)
print("GPU Information:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print("  Running on: GPU ✓")
else:
    print("  Running on: CPU (WARNING: Much slower than GPU)")
print("="*60 + "\n")

# Load model
model = SUIM_Net(base=base_, n_classes=5, pretrained=False)
checkpoint = torch.load(join("SUIM/Pytorch/ckpt/", ckpt_name), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

total_params, trainable_params = model.count_parameters()
print(f"Model: SUIM-Net ({base_})")
print(f"Parameters: {total_params:,}")
print(f"Checkpoint: {ckpt_name}")
print(f"Epoch: {checkpoint['epoch']+1}, Loss: {checkpoint['loss']:.4f}\n")


im_h, im_w = im_res_[1], im_res_[0]

# Create test dataset (no augmentation, no masks needed)
test_dataset = SUIMDataset(
    train_dir="SUIM/TEST",
    image_folder="images",
    mask_folder=None,  # No masks for testing
    target_size=(im_w, im_h),
    augmentation=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Test dataset: {len(test_dataset)} images\n")

def testGenerator():
    # Warm-up run (exclude from timing)
    print("Warming up model...")
    img, _ = test_dataset[0]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warm-up complete.\n")
    
    # Start timing for FPS measurement
    inference_times = []
    start_time = time.time()
    
    with torch.no_grad():
        for idx, (img, img_path) in enumerate(test_loader):
            img = img.to(device)
            
            # inference with timing
            t0 = time.time()
            out_img = model(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU to finish
            inference_time = time.time() - t0
            inference_times.append(inference_time)
            
            # Convert back to numpy: BCHW -> HWC
            out_img = out_img.cpu().numpy()[0].transpose(1, 2, 0)
            
            # thresholding
            out_img[out_img>0.5] = 1.
            out_img[out_img<=0.5] = 0.
            
            # get filename
            img_name = ntpath.basename(img_path[0]).split('.')[0] + '.bmp'
            print(f"Tested [{idx+1}/{len(test_dataset)}]: {ntpath.basename(img_path[0])} ({inference_time*1000:.1f}ms)")
            
            # save individual output masks
            ROs = out_img[:,:,0]
            FVs = out_img[:,:,1]
            HDs = out_img[:,:,2]
            RIs = out_img[:,:,3]
            WRs = out_img[:,:,4]
            Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
            Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
            Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
            Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
            Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)
            
            # Create and save RGB combined mask
            rgb_mask = binaryMasksToRGB(ROs, FVs, HDs, RIs, WRs)
            Image.fromarray(rgb_mask).save(RGB_dir+img_name)
    
    total_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    print("\n" + "="*60)
    print(f"Performance Statistics:")
    print(f"  Total images processed: {len(test_dataset)}")
    print(f"  Total time (with I/O): {total_time:.2f}s")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
    print(f"  FPS (inference only): {fps:.2f}")
    print(f"  Min inference time: {min(inference_times)*1000:.1f}ms")
    print(f"  Max inference time: {max(inference_times)*1000:.1f}ms")
    print("="*60)

# test images
testGenerator()


