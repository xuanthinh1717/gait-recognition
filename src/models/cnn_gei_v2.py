"""
cnn_gei_v2.py

Model GEI (cau hinh D) cho thi nghiem open-set:
    forward(x) -> logits        (de train bang softmax / triplet tren cac nguoi train)
    embed(x)   -> vector dac trung (de so khop cosine luc danh gia open-set)
"""
import torch.nn as nn

try:
    from src.models.cnn_model_attention import CBAMBlock
except ImportError:  # chay doc lap trong cung thu muc
    from cnn_model_attention import CBAMBlock


class GEIEmbedCNN(nn.Module):
    """
    Cau hinh D (CNN+CBAM tren anh GEI):
    Vi chi xu ly 1 anh GEI / video nen giu nguyen luoi khong gian 8x8 (4096 chieu)
    va dung so chieu embedding lon hon (feat_dim, mac dinh 1024 cho CASIA-B, 2048 cho OUMVLP).
    Co BatchNorm2d o 3 khoi Conv va BNNeck o dau ra embed() de tuong thich Cosine / Triplet Loss.
    """

    def __init__(self, num_classes, feat_dim=1024, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 64 -> 32

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 16 -> 8
            CBAMBlock(64, reduction=4),
        )
        fc_in = 64 * 8 * 8                                  # 4096
        hidden_dim = feat_dim * 2
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),                       # BNNeck: chuan hoa cho Cosine / Triplet
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def embed(self, x):                                     # (B, feat_dim)
        return self.proj(self.conv(x))

    def forward(self, x, return_embed=False):               # logits (B, C) hoac (logits, emb)
        emb = self.embed(x)
        logits = self.head(self.drop(emb))
        return (logits, emb) if return_embed else logits
