"""
SUIM-Net with Swin Transformer + Pyramid Pooling Module (PPM)
Adaptado para segmentação subaquática
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


# ---------------------------------------------------------
#                   PYRAMID POOLING MODULE
# ---------------------------------------------------------

class PPM(nn.Module):
    """
    Pyramid Pooling Module (PSPNet)
    Pools input feature map in multiple scales:
    - 1x1  (global context)
    - 2x2
    - 3x3
    - 6x6
    Depois faz upsample e concatena todas as escalas.
    """
    def __init__(self, in_dim, reduction_dim=256, pool_sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, reduction_dim),  # Usar GroupNorm em vez de BatchNorm
                nn.ReLU(inplace=True)
            )
            for ps in pool_sizes
        ])

        # último conv após concatenar todas as escalas
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim + len(pool_sizes) * reduction_dim, in_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, in_dim),  # Usar GroupNorm em vez de BatchNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        ppm_out = [x]

        # aplicar pooling multi-escala
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
            ppm_out.append(y)

        # concat e fusão
        x = torch.cat(ppm_out, dim=1)
        x = self.fusion(x)
        return x


# ---------------------------------------------------------
#              SUIM-Net with Swin Transformer
# ---------------------------------------------------------

class SUIM_Net_Swin_PPM(nn.Module):
    """
    SUIM-Net + Swin Transformer Tiny + PPM Head
    """

    def __init__(self, n_classes=5, pretrained=True, eps=1e-5):
        super().__init__()
        
        # Save eps for BatchNorm layers
        self.eps = eps

        # Load Swin Tiny
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_t(weights=weights)
        self.features = swin.features

        # Feature map dims after each stage
        self.stage_indices = [1, 3, 5, 7]
        self.feature_channels = [96, 192, 384, 768]

        # ================================
        #           PPM HEAD
        # ================================
        self.ppm = PPM(in_dim=768, reduction_dim=128)

        # ================================
        #           DECODER
        # ================================

        self.up1 = self._upsample_block(768, 384)
        self.up2 = self._upsample_block(768, 192)
        self.up3 = self._upsample_block(384, 96)

        self.up4 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=eps),
            nn.ReLU(inplace=True)
        )

        # final upsample (H/4 → H)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=eps),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Conv2d(32, n_classes, kernel_size=1)

    # ---------------------------------------------------------

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c, eps=self.eps),
            nn.ReLU(inplace=True)
        )

    def _permute_features(self, x):
        if len(x.shape) == 4 and x.shape[-1] in self.feature_channels:
            return x.permute(0, 3, 1, 2)
        return x

    def _match_dimensions(self, x, target):
        _, _, h, w = target.shape
        _, _, xh, xw = x.shape

        if xh > h:
            x = x[:, :, :h, :]
        if xw > w:
            x = x[:, :, :, :w]

        if xh < h or xw < w:
            pad_h = h - xh
            pad_w = w - xw
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x

    # ---------------------------------------------------------

    def forward(self, x):
        features = []

        # extrair features do Swin
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.stage_indices:
                features.append(self._permute_features(x))

        feat1, feat2, feat3, feat4 = features  # 96, 192, 384, 768

        # ================================
        #        APPLY PPM TO STAGE 4
        # ================================
        feat4 = self.ppm(feat4)

        # ================================
        #              DECODER
        # ================================

        dec1 = self.up1(feat4)
        dec1 = torch.cat([self._match_dimensions(dec1, feat3), feat3], dim=1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat([self._match_dimensions(dec2, feat2), feat2], dim=1)

        dec3 = self.up3(dec2)
        dec3 = torch.cat([self._match_dimensions(dec3, feat1), feat1], dim=1)

        dec4 = self.up4(dec3)
        dec5 = self.up5(dec4)

        out = self.conv_out(dec5)
        out = torch.sigmoid(out)
        return out

    # ---------------------------------------------------------

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------
#                     TEST
# ---------------------------------------------------------

if __name__ == "__main__":
    model = SUIM_Net_Swin_PPM(n_classes=5, pretrained=True)
    x = torch.randn(1, 3, 256, 320)

    print("Input:", x.shape)
    out = model(x)
    print("Output:", out.shape)

    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} (trainable: {trainable:,})")
