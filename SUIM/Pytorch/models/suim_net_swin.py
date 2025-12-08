"""
# SUIM-Net with Swin Transformer encoder for underwater image segmentation
# Paper: https://arxiv.org/pdf/2004.01241.pdf (SUIM-Net)
# Swin Transformer: https://arxiv.org/abs/2103.14030
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


class SUIM_Net_Swin(nn.Module):
    """
    SUIM-Net with Swin Transformer Tiny encoder
    
    Swin-T feature pyramid:
    - Stage 0: H/4  x W/4  x 96   (after patch partition + linear embedding)
    - Stage 1: H/4  x W/4  x 96   (2 Swin blocks)
    - Stage 2: H/8  x W/8  x 192  (2 Swin blocks + patch merging)
    - Stage 3: H/16 x W/16 x 384  (6 Swin blocks + patch merging)
    - Stage 4: H/32 x W/32 x 768  (2 Swin blocks + patch merging)
    """
    def __init__(self, n_classes=5, pretrained=True):
        super(SUIM_Net_Swin, self).__init__()
        
        # Load Swin Transformer Tiny encoder
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_t(weights=weights)
        
        # Extract encoder features (all stages)
        self.features = swin.features
        
        # Feature dimensions at each stage
        # Stage indices: 0(96), 2(96), 4(192), 6(384), 8(768)
        self.stage_indices = [1, 3, 5, 7]  # After each stage (excluding stage 0)
        self.feature_channels = [96, 192, 384, 768]
        
        # Decoder with skip connections
        # dec-1: 768 → 384 (H/32 → H/16, 2x upsample)
        self.up1 = self._upsample_block(768, 384)
        
        # dec-2: 384 + 384 (skip) → 192 (H/16 → H/8, 2x upsample)
        self.up2 = self._upsample_block(768, 192)  # 768 = 384 + 384
        
        # dec-3: 192 + 192 (skip) → 96 (H/8 → H/4, 2x upsample)
        self.up3 = self._upsample_block(384, 96)   # 384 = 192 + 192
        
        # dec-4: 96 + 96 (skip) → 32 (stays at H/4)
        self.up4 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, padding=1),  # 192 = 96 + 96
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to original resolution (4x upsample from H/4 to H)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.conv_out = nn.Conv2d(32, n_classes, kernel_size=1)
    
    def _upsample_block(self, in_channels, out_channels):
        """
        Upsample block: Upsample 2x + Conv + BN + ReLU
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _permute_features(self, x):
        """
        Convert Swin output (B, H, W, C) to standard format (B, C, H, W)
        """
        if len(x.shape) == 4 and x.shape[-1] in self.feature_channels:
            # Input is (B, H, W, C) -> permute to (B, C, H, W)
            return x.permute(0, 3, 1, 2)
        return x
    
    def forward(self, x):
        """
        Forward pass with feature extraction at multiple scales
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            out: Output segmentation mask (B, n_classes, H, W)
        """
        # Extract features at different stages
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features after each stage (excluding initial patch partition)
            if i in self.stage_indices:
                # Permute if needed: (B, H, W, C) -> (B, C, H, W)
                feat = self._permute_features(x)
                features.append(feat)
        
        # Features: [stage1 (96), stage2 (192), stage3 (384), stage4 (768)]
        feat1, feat2, feat3, feat4 = features
        
        # Decoder with skip connections
        # Stage 4 (768 channels, H/32 x W/32) -> dec1 (384 channels, H/16 x W/16)
        dec1 = self.up1(feat4)
        
        # Match dimensions and concatenate with stage 3 (384 channels)
        dec1 = self._match_dimensions(dec1, feat3)
        dec1 = torch.cat([dec1, feat3], dim=1)  # 768 = 384 + 384
        
        # dec1 (768) -> dec2 (192 channels, H/8 x W/8)
        dec2 = self.up2(dec1)
        
        # Match dimensions and concatenate with stage 2 (192 channels)
        dec2 = self._match_dimensions(dec2, feat2)
        dec2 = torch.cat([dec2, feat2], dim=1)  # 384 = 192 + 192
        
        # dec2 (384) -> dec3 (96 channels, H/4 x W/4)
        dec3 = self.up3(dec2)
        
        # Match dimensions and concatenate with stage 1 (96 channels)
        dec3 = self._match_dimensions(dec3, feat1)
        dec3 = torch.cat([dec3, feat1], dim=1)  # 192 = 96 + 96
        
        # dec3 (192) -> dec4 (96 channels, H/4 x W/4)
        dec4 = self.up4(dec3)
        
        # Final upsampling: H/4 x W/4 -> H x W
        dec5 = self.up5(dec4)
        
        # Output segmentation mask
        out = self.conv_out(dec5)
        out = torch.sigmoid(out)
        
        return out
    
    def _match_dimensions(self, x, target):
        """
        Match spatial dimensions of x to target using padding/cropping
        
        Args:
            x: Tensor to resize (B, C, H, W)
            target: Target tensor with desired spatial dimensions
            
        Returns:
            Resized x matching target's spatial dimensions
        """
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
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Test Swin Transformer model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("SUIM-Net with Swin Transformer Tiny Encoder")
    print("="*60)
    
    model = SUIM_Net_Swin(n_classes=5, pretrained=True).to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 320).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    
    total, trainable = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Size: ~{total/1e6:.1f}M parameters")
    
    # Memory estimation
    print(f"\nMemory (approximate):")
    print(f"  Model: ~{total * 4 / 1e6:.1f} MB (FP32)")
    print(f"  Batch (2 images): ~{2 * 3 * 256 * 320 * 4 / 1e6:.1f} MB")
    
    print("\n" + "="*60)
    print("✓ Model test completed successfully!")
    print("="*60)
