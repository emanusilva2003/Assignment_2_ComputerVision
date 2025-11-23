"""
# Data utility functions for training on the SUIM dataset
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import fnmatch

"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""
def getRobotFishHumanReefWrecks(mask):
    # for categories: HD, RO, FV, WR, RI
    imw, imh = mask.shape[0], mask.shape[1]
    Human = np.zeros((imw, imh))
    Robot = np.zeros((imw, imh))
    Fish = np.zeros((imw, imh))
    Reef = np.zeros((imw, imh))
    Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Human[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                Robot[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                Fish[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Reef[i, j] = 1  
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                Wreck[i, j] = 1  
            else: pass
    return np.stack((Robot, Fish, Human, Reef, Wreck), -1) 


def getRobotFishHumanWrecks(mask):
    # for categories: HD, RO, FV, WR
    imw, imh = mask.shape[0], mask.shape[1]
    Human = np.zeros((imw, imh))
    Robot = np.zeros((imw, imh))
    Fish = np.zeros((imw, imh))
    Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Human[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                Robot[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                Fish[i, j] = 1  
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                Wreck[i, j] = 1  
            else: pass
    return np.stack((Robot, Fish, Human, Wreck), -1) 


def getSaliency(mask):
    # one combined category: HD/RO/FV/WR
    imw, imh = mask.shape[0], mask.shape[1]
    sal = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                sal[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                sal[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                sal[i, j] = 1   
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                sal[i, j] = 0.8  
            else: pass
    return np.expand_dims(sal, axis=-1) 


def processSUIMDataRFHW(img, mask, sal=False):
    # scaling image data and masks
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    m = []
    for i in range(mask.shape[0]):
        if sal:
            m.append(getSaliency(mask[i]))
        else:
            m.append(getRobotFishHumanReefWrecks(mask[i]))
            #m.append(getRobotFishHumanWrecks(mask[i]))
    m = np.array(m)
    return (img, m)


def trainDataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", target_size=(256,256), sal=False):
    # data generator function for driving the training
    from tensorflow.keras.preprocessing import image as keras_image
    import glob
    
    # Get all image paths
    image_dir = os.path.join(train_path, image_folder)
    mask_dir = os.path.join(train_path, mask_folder)
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}. Check if the path is correct.")
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    idx = 0
    while True:
        batch_images = []
        batch_masks = []
        
        for _ in range(batch_size):
            if idx >= len(image_paths):
                idx = 0
                
            img_path = image_paths[idx]
            img_name = os.path.basename(img_path)
            # Change extension to .bmp for mask
            mask_name = os.path.splitext(img_name)[0] + '.bmp'
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Load image
            if image_color_mode == "rgb":
                img = keras_image.load_img(img_path, target_size=target_size, color_mode='rgb')
            else:
                img = keras_image.load_img(img_path, target_size=target_size, color_mode='grayscale')
            img = keras_image.img_to_array(img)
            
            # Load mask
            if mask_color_mode == "rgb":
                mask = keras_image.load_img(mask_path, target_size=target_size, color_mode='rgb')
            else:
                mask = keras_image.load_img(mask_path, target_size=target_size, color_mode='grayscale')
            mask = keras_image.img_to_array(mask)
            
            batch_images.append(img)
            batch_masks.append(mask)
            idx += 1
            
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        
        # Apply augmentation with same seed
        seed = np.random.randint(0, 10000)
        batch_images = next(image_datagen.flow(batch_images, batch_size=batch_size, seed=seed, shuffle=False))
        batch_masks = next(mask_datagen.flow(batch_masks, batch_size=batch_size, seed=seed, shuffle=False))
        
        # Process the data
        batch_images, batch_masks = processSUIMDataRFHW(batch_images, batch_masks, sal)
        
        yield (batch_images, batch_masks)


def getPaths(data_dir):
    # read image files from directory
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return image_paths


