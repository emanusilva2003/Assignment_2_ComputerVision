
import os
import fnmatch
import numpy as np
from PIL import Image

"""
RGB color code and object categories:    Cada pixel da máscara é uma cor RGB, mas aqui os canais são 0 ou 1
------------------------------------
000 BW: Background waterbody (não incluído - tudo que não é classe)
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""


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
    return sorted(list(set(image_paths)))


def binaryMasksToRGB(masks, num_classes=5):
    """
    Convert binary masks to a single RGB image using the SUIM color encoding
    
    Args:
        masks: numpy array or torch tensor
            - For 5 classes: (5, H, W) with order [RO, FV, HD, RI, WR]
            - For 7 classes: (7, H, W) with order [HD, PF, WR, RO, RI, FV, SR]
        num_classes: Number of classes (5 or 7)
    
    Returns:
        RGB image as numpy array (H, W, 3) with uint8 values
    
    Color encoding:
        5 classes:
            RO (Robots) - Red (255, 0, 0)
            FV (Fish/vertebrates) - Yellow (255, 255, 0)
            HD (Human divers) - Blue (0, 0, 255)
            RI (Reefs/invertebrates) - Magenta (255, 0, 255)
            WR (Wrecks/ruins) - Cyan (0, 255, 255)
        7 classes:
            HD (Human divers) - Blue (0, 0, 255)
            PF (Plants/sea-grass) - Green (0, 255, 0)
            WR (Wrecks/ruins) - Cyan (0, 255, 255)
            RO (Robots) - Red (255, 0, 0)
            RI (Reefs/invertebrates) - Magenta (255, 0, 255)
            FV (Fish/vertebrates) - Yellow (255, 255, 0)
            SR (Sand/sea-floor) - White (255, 255, 255)
        Background - Black (0, 0, 0)
    """
    # Convert to numpy if torch tensor
    if hasattr(masks, 'cpu'):
        masks = masks.cpu().numpy()
    
    if num_classes == 5:
        # Order: RO, FV, HD, RI, WR
        RO, FV, HD, RI, WR = masks[0], masks[1], masks[2], masks[3], masks[4]
        h, w = RO.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors with priority (later ones override earlier ones where they overlap)
        rgb_mask[RO > 0] = [255, 0, 0]      # Red
        rgb_mask[FV > 0] = [255, 255, 0]    # Yellow
        rgb_mask[HD > 0] = [0, 0, 255]      # Blue
        rgb_mask[RI > 0] = [255, 0, 255]    # Magenta
        rgb_mask[WR > 0] = [0, 255, 255]    # Cyan
        
        return rgb_mask
    
    elif num_classes == 7:
        # Order: HD, PF, WR, RO, RI, FV, SR
        HD, PF, WR, RO, RI, FV, SR = masks[0], masks[1], masks[2], masks[3], masks[4], masks[5], masks[6]
        h, w = HD.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors with priority (later ones override earlier ones where they overlap)
        rgb_mask[HD > 0] = [0, 0, 255]      # Blue
        rgb_mask[PF > 0] = [0, 255, 0]      # Green
        rgb_mask[WR > 0] = [0, 255, 255]    # Cyan
        rgb_mask[RO > 0] = [255, 0, 0]      # Red
        rgb_mask[RI > 0] = [255, 0, 255]    # Magenta
        rgb_mask[FV > 0] = [255, 255, 0]    # Yellow
        rgb_mask[SR > 0] = [255, 255, 255]  # White
        
        return rgb_mask
    
    else:
        raise ValueError(f"num_classes must be 5 or 7, got {num_classes}")


