"""
# Data utility functions for training SUIM-Net in PyTorch
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

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
    """
    Extract 5 binary masks from RGB mask
    Categories: RO, FV, HD, RI, WR
    """
    imw, imh = mask.shape[0], mask.shape[1]
    Human = np.zeros((imw, imh), dtype=np.float32)
    Robot = np.zeros((imw, imh), dtype=np.float32)
    Fish = np.zeros((imw, imh), dtype=np.float32)
    Reef = np.zeros((imw, imh), dtype=np.float32)
    Wreck = np.zeros((imw, imh), dtype=np.float32)
    
    for i in range(imw):
        for j in range(imh):
            if mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1:
                Human[i, j] = 1.0
            elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0:
                Robot[i, j] = 1.0
            elif mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0:
                Fish[i, j] = 1.0
            elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1:
                Reef[i, j] = 1.0
            elif mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1:
                Wreck[i, j] = 1.0
    
    # Stack as (5, H, W) for PyTorch
    return np.stack((Robot, Fish, Human, Reef, Wreck), axis=0)


class SUIMDataset(Dataset):
    """
    PyTorch Dataset for SUIM with data augmentation
    """
    def __init__(self, train_dir, image_folder="images", mask_folder="masks", 
                 target_size=(320, 256), augmentation=True, augmentation_params=None):
        """
        Args:
            train_dir: Root directory containing image_folder and mask_folder
            image_folder: Subfolder name for images
            mask_folder: Subfolder name for masks
            target_size: (width, height) tuple
            augmentation: Whether to apply data augmentation
            augmentation_params: Dict with augmentation parameters (rotation, shift, etc.)
        """
        self.train_dir = train_dir
        self.image_dir = os.path.join(train_dir, image_folder)
        self.mask_dir = os.path.join(train_dir, mask_folder)
        self.target_size = target_size
        self.augmentation = augmentation
        
        # Get all image paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.*')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Default augmentation parameters (matching Keras config)
        if augmentation_params is None:
            augmentation_params = {
                'rotation_range': 0.2,
                'width_shift_range': 0.05,
                'height_shift_range': 0.05,
                'shear_range': 0.05,
                'zoom_range': 0.05,
                'horizontal_flip': True
            }
        self.aug_params = augmentation_params
        
        # Basic transforms (always applied)
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        mask_name = os.path.splitext(img_name)[0] + '.bmp'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load as PIL images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # Resize
        """
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        """
        # Apply augmentation if enabled
        if self.augmentation:
            image, mask = self._apply_augmentation(image, mask)
        
        # Convert to tensors
        image = self.to_tensor(image)  # (3, H, W), values in [0, 1]
        
        # Normazalize image to [-1, 1]
        # ver + tarde 
        

        # Process mask: RGB -> 5 binary channels
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np[mask_np > 0.5] = 1.0
        mask_np[mask_np <= 0.5] = 0.0
        
        # Extract 5 categories
        mask_tensor = getRobotFishHumanReefWrecks(mask_np)  # (5, H, W)
        mask_tensor = torch.from_numpy(mask_tensor)
        
        return image, mask_tensor
    
    def _apply_augmentation(self, image, mask):
        """
        Apply synchronized augmentation to image and mask
        Mimics Keras ImageDataGenerator behavior
        """
        # Random horizontal flip
        if self.aug_params.get('horizontal_flip', False) and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation (in degrees)
        if self.aug_params.get('rotation_range', 0) > 0:
            angle = random.uniform(-self.aug_params['rotation_range'] * 180, 
                                   self.aug_params['rotation_range'] * 180)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        # Random zoom (scale)
        if self.aug_params.get('zoom_range', 0) > 0:
            zoom_factor = 1.0 + random.uniform(-self.aug_params['zoom_range'], 
                                               self.aug_params['zoom_range'])
            new_size = (int(image.size[0] * zoom_factor), int(image.size[1] * zoom_factor))
            image = TF.resize(image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, new_size, interpolation=transforms.InterpolationMode.NEAREST)
            # Crop back to original size
            image = TF.center_crop(image, self.target_size[::-1])  # (H, W)
            mask = TF.center_crop(mask, self.target_size[::-1])
        
        # Random shift (translate)
        shift_x = shift_y = 0
        if self.aug_params.get('width_shift_range', 0) > 0:
            shift_x = int(random.uniform(-self.aug_params['width_shift_range'], 
                                         self.aug_params['width_shift_range']) * image.size[0])
        if self.aug_params.get('height_shift_range', 0) > 0:
            shift_y = int(random.uniform(-self.aug_params['height_shift_range'], 
                                         self.aug_params['height_shift_range']) * image.size[1])
        if shift_x != 0 or shift_y != 0:
            image = TF.affine(image, angle=0, translate=[shift_x, shift_y], scale=1.0, shear=0,
                             interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[shift_x, shift_y], scale=1.0, shear=0,
                            interpolation=transforms.InterpolationMode.NEAREST)
        
        # Random shear
        if self.aug_params.get('shear_range', 0) > 0:
            shear = random.uniform(-self.aug_params['shear_range'] * 180, 
                                   self.aug_params['shear_range'] * 180)
            image = TF.affine(image, angle=0, translate=[0, 0], scale=1.0, shear=shear,
                             interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[0, 0], scale=1.0, shear=shear,
                            interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, mask


def get_suim_dataloader(train_dir, batch_size=8, image_folder="images", mask_folder="masks",
                        target_size=(320, 240), augmentation=True, augmentation_params=None,
                        num_workers=4, shuffle=True):
    """
    Create PyTorch DataLoader for SUIM dataset
    
    Args:
        train_dir: Root directory containing images and masks
        batch_size: Batch size
        image_folder: Subfolder for images
        mask_folder: Subfolder for masks
        target_size: (width, height) tuple
        augmentation: Whether to apply augmentation
        augmentation_params: Dict with augmentation parameters
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader object
    """
    dataset = SUIMDataset(train_dir, image_folder, mask_folder, 
                          target_size, augmentation, augmentation_params)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=True)
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    train_dir = "SUIM/train_val"
    
    # Create dataset
    dataset = SUIMDataset(train_dir, target_size=(320, 240), augmentation=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}")  # Should be (3, 240, 320)
    print(f"Mask shape: {mask.shape}")   # Should be (5, 240, 320)
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Test dataloader
    dataloader = get_suim_dataloader(train_dir, batch_size=4, num_workers=0)
    for imgs, masks in dataloader:
        print(f"Batch - Images: {imgs.shape}, Masks: {masks.shape}")
        break
