"""
# SUIM-Net model for underwater image segmentation (PyTorch implementation)
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class RSB(nn.Module):
    """
    Residual Skip Block
    Implements the bottleneck residual block: 1x1 -> 3x3 -> 1x1 + skip
    """
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, skip=True):
        super(RSB, self).__init__()
        f1, f2, f3, f4 = filters
        self.skip = skip
        
        # Sub-block 1: 1x1 conv (bottleneck)
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.2)  # PyTorch uses (1-momentum)
        
        # Sub-block 2: 3x3 conv (feature extraction)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.2)
        
        # Sub-block 3: 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(f3, momentum=0.2)
        
        # Skip connection (shortcut)
        if not skip:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, f4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(f4, momentum=0.2)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        identity = self.shortcut(identity)
        
        # Add and activate
        out = out + identity
        out = self.relu(out)
        
        return out


class SUIM_Encoder_RSB(nn.Module):
    """
    SUIM-Net Encoder with RSB blocks
    """
    def __init__(self, in_channels=3):
        super(SUIM_Encoder_RSB, self).__init__()
        
        # Encoder block 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False)  # padding=2 to maintain size
        
        # Encoder block 2
        self.bn2 = nn.BatchNorm2d(64, momentum=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.rsb2_1 = RSB(64, [64, 64, 128, 128], stride=2, skip=False)
        self.rsb2_2 = RSB(128, [64, 64, 128, 128], skip=True)
        self.rsb2_3 = RSB(128, [64, 64, 128, 128], skip=True)
        
        # Encoder block 3
        self.rsb3_1 = RSB(128, [128, 128, 256, 256], stride=2, skip=False)
        self.rsb3_2 = RSB(256, [128, 128, 256, 256], skip=True)
        self.rsb3_3 = RSB(256, [128, 128, 256, 256], skip=True)
        self.rsb3_4 = RSB(256, [128, 128, 256, 256], skip=True)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder block 1
        enc1 = self.conv1(x)
        
        # Encoder block 2
        x = self.bn2(enc1)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.rsb2_1(x)
        x = self.rsb2_2(x)
        enc2 = self.rsb2_3(x)
        
        # Encoder block 3
        x = self.rsb3_1(enc2)
        x = self.rsb3_2(x)
        x = self.rsb3_3(x)
        enc3 = self.rsb3_4(x)
        
        return enc1, enc2, enc3


class SUIM_Decoder_RSB(nn.Module):
    """
    SUIM-Net Decoder with skip connections
    """
    def __init__(self, n_classes=5):
        super(SUIM_Decoder_RSB, self).__init__()
        
        # Decoder block 1
        self.conv_dec1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(256, momentum=0.2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_skip1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # enc2 has 128 channels
        self.bn_skip1 = nn.BatchNorm2d(256, momentum=0.2)
        
        # Decoder block 2
        self.conv_dec2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 + 256 (concat)
        self.bn_dec2 = nn.BatchNorm2d(256, momentum=0.2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_dec2s = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_dec2s = nn.BatchNorm2d(128, momentum=0.2)
        self.upsample2s = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_skip2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # enc1 has 64 channels
        self.bn_skip2 = nn.BatchNorm2d(128, momentum=0.2)
        
        # Decoder block 3
        self.conv_dec3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 = 128 + 128 (concat)
        self.bn_dec3 = nn.BatchNorm2d(128, momentum=0.2)
        self.conv_dec3s = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_dec3s = nn.BatchNorm2d(64, momentum=0.2)
        
        # Output layer
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, enc1, enc2, enc3):
        # Decoder block 1
        dec1 = self.conv_dec1(enc3)
        dec1 = self.bn_dec1(dec1)
        dec1 = self.upsample1(dec1)
        
        # Match dimensions with padding/cropping
        dec1 = self._match_dimensions(dec1, enc2)
        
        # Skip connection 1
        skip1 = self.conv_skip1(enc2)
        skip1 = self.bn_skip1(skip1)
        skip1 = self.relu(skip1)
        dec1s = torch.cat([skip1, dec1], dim=1)
        
        # Decoder block 2
        dec2 = self.conv_dec2(dec1s)
        dec2 = self.bn_dec2(dec2)
        dec2 = self.upsample2(dec2)
        
        dec2s = self.conv_dec2s(dec2)
        dec2s = self.bn_dec2s(dec2s)
        dec2s = self.upsample2s(dec2s)
        
        # Match dimensions
        dec2s = self._match_dimensions(dec2s, enc1)
        
        # Skip connection 2
        skip2 = self.conv_skip2(enc1)
        skip2 = self.bn_skip2(skip2)
        skip2 = self.relu(skip2)
        dec2s_cat = torch.cat([skip2, dec2s], dim=1)
        
        # Decoder block 3
        dec3 = self.conv_dec3(dec2s_cat)
        dec3 = self.bn_dec3(dec3)
        dec3s = self.conv_dec3s(dec3)
        dec3s = self.bn_dec3s(dec3s)
        
        # Output
        out = self.conv_out(dec3s)
        out = torch.sigmoid(out)
        
        return out
    
    def _match_dimensions(self, x, target):
        """Match spatial dimensions of x to target"""
        _, _, h, w = target.shape
        _, _, x_h, x_w = x.shape
        
        # Crop if larger
        if x_h > h:
            x = x[:, :, :h, :]
        if x_w > w:
            x = x[:, :, :, :w]
        
        # Pad if smaller
        if x_h < h or x_w < w:
            pad_h = h - x_h
            pad_w = w - x_w
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        return x


class SUIM_Net_VGG(nn.Module):
    """
    SUIM-Net with VGG16 encoder
    """
    def __init__(self, n_classes=5, pretrained=True):
        super(SUIM_Net_VGG, self).__init__()
        
        # Load VGG16 encoder
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        self.features = vgg.features
        
        # Extract specific pooling layers
        self.pool1_idx = 4   # block1_pool (64 filters)
        self.pool2_idx = 9   # block2_pool (128 filters)
        self.pool3_idx = 16  # block3_pool (256 filters)
        self.pool4_idx = 23  # block4_pool (512 filters)
        
        # Decoder (matching Keras implementation: 512, 256, 128)
        # Paper architecture uses: 768, 384, 192 (but Keras uses 512, 256, 128)
        self.up1 = self._upsample_block(512, 512, 512)  # enc-4 (512) → dec-1 (512) [Paper: 768]
        self.up2 = self._upsample_block(768, 256, 256)  # 768 = 512 + 256 (dec-1 + enc-3) [Paper: 384]
        self.up3 = self._upsample_block(384, 128, 128)  # 384 = 256 + 128 (dec-2 + enc-2) [Paper: 192]
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output
        self.conv_out = nn.Conv2d(192, n_classes, kernel_size=3, padding=1)  # 192 = 128 + 64 (dec-3 + enc-1)
    
    def _upsample_block(self, in_channels, out_channels, skip_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract features at different pooling layers
        pools = []
        for i, layer in enumerate(self.features):
            x = layer(x) # Each VGG layer applied sequentially
            if i in [self.pool1_idx, self.pool2_idx, self.pool3_idx, self.pool4_idx]:
                pools.append(x)
        
        pool1, pool2, pool3, pool4 = pools
        
        # Decoder with skip connections
        dec1 = self.up1(pool4)
        dec1 = torch.cat([dec1, pool3], dim=1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, pool2], dim=1)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat([dec3, pool1], dim=1)
        
        dec4 = self.up4(dec3)
        
        # Output
        out = self.conv_out(dec4)
        out = torch.sigmoid(out)
        
        return out


class SUIM_Net(nn.Module):
    """
    SUIM-Net model wrapper
    - base='RSB' for RSB-based encoder
    - base='VGG' for VGG16 encoder
    """
    def __init__(self, base='RSB', n_classes=5, pretrained=True):
        super(SUIM_Net, self).__init__()
        
        if base == 'RSB':
            self.encoder = SUIM_Encoder_RSB(in_channels=3)
            self.decoder = SUIM_Decoder_RSB(n_classes=n_classes)
            self.base = 'RSB'
        elif base == 'VGG':
            self.model = SUIM_Net_VGG(n_classes=n_classes, pretrained=pretrained)
            self.base = 'VGG'
        else:
            raise ValueError(f"Unknown base: {base}. Use 'RSB' or 'VGG'")
    
    def forward(self, x):
        if self.base == 'RSB':
            enc1, enc2, enc3 = self.encoder(x)
            out = self.decoder(enc1, enc2, enc3)
        else:  # VGG
            out = self.model(x)
        return out
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Test RSB model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_rsb = SUIM_Net(base='RSB', n_classes=5).to(device)
    
    # Test VGG model
    model_vgg = SUIM_Net(base='VGG', n_classes=5).to(device)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 320).to(device)
    
    print("RSB Model:")
    out_rsb = model_rsb(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_rsb.shape}")
    total, trainable = model_rsb.count_parameters()
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    
    print("\nVGG Model:")
    out_vgg = model_vgg(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_vgg.shape}")
    total, trainable = model_vgg.count_parameters()
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
