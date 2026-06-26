import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Tách MLP riêng, không dùng Flatten trong Sequential
        # để tránh shape conflict khi share weights giữa avg và max
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        # Flatten thủ công trước khi đưa vào MLP
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))

        # Kết hợp avg + max rồi scale lại feature map gốc
        scale = self.sigmoid(avg_out + max_out)
        scale = scale.view(b, c, 1, 1)

        return x * scale


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,       # avg + max theo channel dimension
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool theo channel axis để lấy spatial context
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        concat = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        scale = self.sigmoid(self.conv(concat))          # (B, 1, H, W)

        return x * scale


class CBAMBlock(nn.Module):

    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        residual = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x + residual


class CNN(nn.Module):

    def __init__(
        self,
        num_classes,
        num_conv_blocks=3,
        dropout=0.3
    ):
        super().__init__()

        assert num_conv_blocks in (2, 3), "num_conv_blocks phải là 2 hoặc 3"

        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # --- Block 2 + CBAM (luôn có) ---
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        if num_conv_blocks == 2:
            self.cbam2 = CBAMBlock(32)
            self.block3 = None
            self.cbam3 = None
            fc_in = 32 * 16 * 16
        else:
            self.cbam2 = None
            self.block3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                # Không MaxPool để giữ spatial resolution cho spatial attention
            )
            self.cbam3 = CBAMBlock(64)
            fc_in = 64 * 16 * 16

        # --- Classifier ---
        layers = [
            nn.Flatten(),
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(512, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        if self.cbam2 is not None:
            x = self.cbam2(x)

        if self.block3 is not None:
            x = self.block3(x)
            x = self.cbam3(x)

        x = self.classifier(x)
        return x
