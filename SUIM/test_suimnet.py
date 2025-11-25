"""
# Test script for the SUIM-Net
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
import time
import tensorflow as tf
# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import getPaths, binaryMasksToRGB

## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "SUIM/TEST/images/"

## sample and ckpt dir
samples_dir = "SUIM/TEST/output/"
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

## input/output shapes
base_ = 'VGG' # or 'RSB'
if base_=='RSB':
    im_res_ = (320, 240, 3) 
    ckpt_name = "suimnet_rsb5.hdf5"
else: 
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg5.hdf5"
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)
model = suimnet.model

print("\n" + "="*60)
print("GPU Information:")
print(f"  TensorFlow version: {tf.__version__}")
print(f"  GPU available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print(f"  GPU devices: {[gpu.name for gpu in tf.config.list_physical_devices('GPU')]}")
    print("  Running on: GPU ✓")
else:
    print("  Running on: CPU (WARNING: Much slower than GPU)")
print("="*60 + "\n")

model.summary()
model.load_weights(join("SUIM/ckpt/", ckpt_name))


im_h, im_w = im_res_[1], im_res_[0]
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    
    image_paths = getPaths(test_dir)
    num_images = len(image_paths)
    
    # Warm-up run (exclude from timing)
    print("Warming up model...")
    img = Image.open(image_paths[0]).resize((im_w, im_h))
    img = np.array(img)/255.
    img = np.expand_dims(img, axis=0)
    _ = model.predict(img, verbose=0)
    print("Warm-up complete.\n")
    
    # Start timing for FPS measurement
    inference_times = []
    start_time = time.time()
    
    for idx, p in enumerate(image_paths):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        img = np.expand_dims(img, axis=0)
        
        # inference with timing
        t0 = time.time()
        out_img = model.predict(img, verbose=0)
        inference_time = time.time() - t0
        inference_times.append(inference_time)
        
        # thresholding
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        print(f"Tested [{idx+1}/{num_images}]: {ntpath.basename(p)} ({inference_time*1000:.1f}ms)")
        
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
        FVs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
        HDs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
        RIs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
        WRs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
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
    print(f"  Total images processed: {num_images}")
    print(f"  Total time (with I/O): {total_time:.2f}s")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
    print(f"  FPS (inference only): {fps:.2f}")
    print(f"  Min inference time: {min(inference_times)*1000:.1f}ms")
    print(f"  Max inference time: {max(inference_times)*1000:.1f}ms")
    print("="*60)

# test images
testGenerator()


