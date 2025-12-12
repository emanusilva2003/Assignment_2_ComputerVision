
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import os
import fnmatch

def getRobotFishHumanReefWrecks(mask, num_classes=5):
    """
    Extract binary masks from RGB mask
    
    Args:
        mask: RGB mask array (H, W, 3) with values 0 or 1
        num_classes: Number of classes to extract (5 or 7)
            - 5 classes: RO, FV, HD, RI, WR (original SUIM-Net)
            - 7 classes: HD, PF, WR, RO, RI, FV, SR (all categories except background)
    
    Returns:
        numpy array of shape (num_classes, H, W)
    """
    imw, imh = mask.shape[0], mask.shape[1]
    
    if num_classes == 5:
        # Original 5 classes: RO, FV, HD, RI, WR
        Human = np.zeros((imw, imh), dtype=np.float32)
        Robot = np.zeros((imw, imh), dtype=np.float32)
        Fish = np.zeros((imw, imh), dtype=np.float32)
        Reef = np.zeros((imw, imh), dtype=np.float32)
        Wreck = np.zeros((imw, imh), dtype=np.float32)
        
        for i in range(imw):
            for j in range(imh):
                if mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1:      # Blue (001)
                    Human[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0:    # Red (100)
                    Robot[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0:    # Yellow (110)
                    Fish[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1:    # Magenta (101)
                    Reef[i, j] = 1.0
                elif mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1:    # Cyan (011)
                    Wreck[i, j] = 1.0
        
        # Stack as (5, H, W) for PyTorch: RO, FV, HD, RI, WR
        return np.stack((Robot, Fish, Human, Reef, Wreck), axis=0)
    
    elif num_classes == 7:
        # All 7 classes (excluding background): HD, PF, WR, RO, RI, FV, SR
        Human = np.zeros((imw, imh), dtype=np.float32)
        Plants = np.zeros((imw, imh), dtype=np.float32)
        Wreck = np.zeros((imw, imh), dtype=np.float32)
        Robot = np.zeros((imw, imh), dtype=np.float32)
        Reef = np.zeros((imw, imh), dtype=np.float32)
        Fish = np.zeros((imw, imh), dtype=np.float32)
        Sand = np.zeros((imw, imh), dtype=np.float32)
        
        for i in range(imw):
            for j in range(imh):
                if mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1:      # Blue (001) - Human
                    Human[i, j] = 1.0
                elif mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==0:    # Green (010) - Plants
                    Plants[i, j] = 1.0
                elif mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1:    # Cyan (011) - Wreck
                    Wreck[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0:    # Red (100) - Robot
                    Robot[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1:    # Magenta (101) - Reef
                    Reef[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0:    # Yellow (110) - Fish
                    Fish[i, j] = 1.0
                elif mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==1:    # White (111) - Sand
                    Sand[i, j] = 1.0
                # Black (000) is background - not included
        
        # Stack as (7, H, W) for PyTorch: HD, PF, WR, RO, RI, FV, SR
        return np.stack((Human, Plants, Wreck, Robot, Reef, Fish, Sand), axis=0)
    
    else:
        raise ValueError(f"num_classes must be 5 or 7, got {num_classes}")

class SUIMDataset(Dataset):
    """
    PyTorch Dataset for SUIM with data augmentation
    """
    def __init__(self, train_dir, image_folder="images", mask_folder="masks", 
                 target_size=(320, 256), augmentation=True, augmentation_params=None, num_classes=5):
        """
        Args:
            train_dir: Root directory containing image_folder and mask_folder
            image_folder: Subfolder name for images
            mask_folder: Subfolder name for masks
            target_size: (width, height) tuple
            augmentation: Whether to apply data augmentation
            augmentation_params: Dict with augmentation parameters (rotation, shift, etc.)
            num_classes: Number of classes to extract from masks (5 or 7)
                - 5: RO, FV, HD, RI, WR (original SUIM-Net)
                - 7: HD, PF, WR, RO, RI, FV, SR (all categories except background)
        """
        self.train_dir = train_dir
        self.image_dir = os.path.join(train_dir, image_folder)
        if mask_folder != None:
            self.mask_dir = os.path.join(train_dir, mask_folder)
        else:
            self.mask_dir = None
        self.target_size = target_size                          # (width, height) para o qual as imagens vão ser cortadas
        self.augmentation = augmentation
        self.num_classes = num_classes
        
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
        return len(self.image_paths) # Total number of images
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]       
        img_name = os.path.basename(img_path)  
        
        # Load as PIL images
        image = Image.open(img_path).convert('RGB')
        
        # Check if mask_folder is provided (training mode)
        if self.mask_dir and os.path.exists(self.mask_dir):
            mask_name = os.path.splitext(img_name)[0] + '.bmp'      
            mask_path = os.path.join(self.mask_dir, mask_name)     # Corresponding mask path
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
                
            mask = Image.open(mask_path).convert('RGB')
            
            # Apply augmentation if enabled
            if self.augmentation:
                image, mask = self._apply_augmentation(image, mask)
            
            # Ensure target size (after augmentation might change size slightly)
            if image.size != self.target_size:
                image = image.resize(self.target_size, Image.BILINEAR)
                mask = mask.resize(self.target_size, Image.NEAREST)
            
            # Convert to tensors
            image = self.to_tensor(image)  # (3, H, W), values in [0, 1]
            
            # Process mask: RGB -> binary channels (5 or 7 depending on num_classes)
            mask_np = np.array(mask).astype(np.float32) / 255.0
            mask_np[mask_np > 0.5] = 1.0
            mask_np[mask_np <= 0.5] = 0.0
            
            # Extract categories (5 or 7)
            mask_tensor = getRobotFishHumanReefWrecks(mask_np, num_classes=self.num_classes)  # (num_classes, H, W)
            mask_tensor = torch.from_numpy(mask_tensor)
            
            return image, mask_tensor
        else:
            # Test mode: return only image and path
            # Resize to target size
            image = image.resize(self.target_size, Image.BILINEAR)
            image = self.to_tensor(image)  # (3, H, W), values in [0, 1]
            
            return image, img_path
    
    def _apply_augmentation(self, image, mask):
        """
        Apply synchronized augmentation to image and mask
        Mimics Keras ImageDataGenerator behavior
        """
        # Random horizontal flip (dict.get(key, fallback))  erve para duplicar dados virando tudo de um lado para o outro.
        if self.aug_params.get('horizontal_flip', False) and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation (in degrees)
        if self.aug_params.get('rotation_range', 0) > 0:
            angle = random.uniform(-self.aug_params['rotation_range'] * 180, 
                                   self.aug_params['rotation_range'] * 180)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        # Random zoom (scale)            Redimensionar a imagem e a máscara para esse tamanho maior/menor
        if self.aug_params.get('zoom_range', 0) > 0:
            zoom_factor = 1.0 + random.uniform(-self.aug_params['zoom_range'], 
                                               self.aug_params['zoom_range'])
            new_size = (int(image.size[0] * zoom_factor), int(image.size[1] * zoom_factor))
            image = TF.resize(image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, new_size, interpolation=transforms.InterpolationMode.NEAREST)
            # Crop back to original size
            image = TF.center_crop(image, self.target_size[::-1])  # (H, W)
            mask = TF.center_crop(mask, self.target_size[::-1])
        
        # Random shift (translate)    Isto move a imagem para cima/baixo/esquerda/direita
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
        
        # Random shear         inclina a imagem na diagonal
        if self.aug_params.get('shear_range', 0) > 0:
            shear = random.uniform(-self.aug_params['shear_range'] * 180, 
                                   self.aug_params['shear_range'] * 180)
            image = TF.affine(image, angle=0, translate=[0, 0], scale=1.0, shear=shear,
                             interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[0, 0], scale=1.0, shear=shear,
                            interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, mask


def get_suim_dataloader(train_dir, batch_size=8, image_folder="images", mask_folder="masks",
                        target_size=(320, 256), augmentation=True, augmentation_params=None,
                        num_workers=4, shuffle=True, validation=False, val_ratio=0.2, drop_last=False, num_classes=5):
    """
    Create PyTorch DataLoader for SUIM dataset
    
    Args:
        train_dir: Root directory containing images and masks
        batch_size: Batch size
        image_folder: Subfolder for images
        mask_folder: Subfolder for masks
        target_size: (width, height) tuple
        augmentation: Whether to apply augmentation (only for training set)
        augmentation_params: Dict with augmentation parameters
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        validation: If True, split dataset into train/val and return both loaders
        val_ratio: Fraction of data to use for validation (default: 0.2 = 20%)
        num_classes: Number of classes to extract from masks (5 or 7)
    
    Returns:
        DataLoader object, or (train_loader, val_loader) if validation=True
    """
    
    if validation:
        # Create train dataset with augmentation
        train_dataset = SUIMDataset(train_dir, image_folder, mask_folder, 
                                    target_size, augmentation=augmentation, 
                                    augmentation_params=augmentation_params,
                                    num_classes=num_classes)
        
        # Create validation dataset without augmentation
        val_dataset = SUIMDataset(train_dir, image_folder, mask_folder, 
                                  target_size, augmentation=False, 
                                  augmentation_params=None,
                                  num_classes=num_classes)
        
        # Split both datasets with same indices
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size
        
        # Get split indices
        indices = list(range(dataset_size))
        generator = torch.Generator().manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        train_indices = torch.randperm(dataset_size, generator=generator)[:train_size].tolist()
        val_indices = torch.randperm(dataset_size, generator=generator)[train_size:].tolist()
        
        # Create subsets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Create train loader (with augmentation and shuffle)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        
        # Create validation loader (no augmentation, no shuffle)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        
        return train_loader, val_loader
    else:
        # Single dataset (original behavior)
        dataset = SUIMDataset(train_dir, image_folder, mask_folder, 
                            target_size, augmentation, augmentation_params, num_classes=num_classes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        return dataloader


if __name__ == "__main__":
    # Test the dataset
    train_dir = "SUIM/train_val"
    
    # Create dataset
    dataset = SUIMDataset(train_dir, target_size=(320, 256), augmentation=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}")  # Should be (3, 256, 320)
    print(f"Mask shape: {mask.shape}")   # Should be (5, 256, 320)
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Test dataloader
    dataloader = get_suim_dataloader(train_dir, batch_size=4, num_workers=0)
    for imgs, masks in dataloader:
        print(f"Batch - Images: {imgs.shape}, Masks: {masks.shape}")
        break
