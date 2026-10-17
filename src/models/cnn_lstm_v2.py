"""
cnn_lstm_v2.py

Cac model cho thi nghiem open-set. Moi model co:
    forward(x) -> logits        (de train bang softmax tren cac nguoi train)
    embed(x)   -> vector dac trung (de so khop cosine luc danh gia open-set)

    CNNLSTM(temporal="lstm") : cau hinh B  (CNN tung frame -> LSTM -> trung binh theo T)
    CNNLSTM(temporal="mean") : cau hinh E  (CNN tung frame -> trung binh theo T, KHONG LSTM)
    GEIEmbedCNN              : cau hinh D  (CNN+CBAM hien tai tren GEI, chi them embed())

B va E dung CHUNG FrameEncoder, chi khac cach gop thoi gian.
"""
import torch.nn as nn

try:
    from src.models.cnn_model_attention import CNN as _GEICNN, CBAMBlock
except ImportError:  # chay doc lap trong cung thu muc
    from cnn_model_attention import CNN as _GEICNN, CBAMBlock


class FrameEncoder(nn.Module):
    """Giong 3 khoi conv + CBAM(64) cua model hien tai, nhung GAP + FC thay cho Flatten."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            CBAMBlock(64),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):                        # (N, 1, S, S)
        x = self.gap(self.conv(x)).flatten(1)    # (N, 64)
        return self.fc(x)                        # (N, out_dim)


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, temporal="lstm", feat_dim=256, dropout=0.3):
        super().__init__()
        assert temporal in ("lstm", "mean")
        self.temporal = temporal
        self.encoder = FrameEncoder(feat_dim)
        if temporal == "lstm":
            # hidden = feat_dim de B va E co cung so chieu embedding (256)
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def embed(self, x):                          # x: (B, T, 1, S, S)
        B, T = x.shape[:2]
        f = self.encoder(x.flatten(0, 1)).view(B, T, -1)   # (B, T, feat_dim)
        if self.temporal == "lstm":
            f, _ = self.lstm(f)                            # (B, T, feat_dim)
        return f.mean(dim=1)                               # (B, feat_dim)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


class GEIEmbedCNN(_GEICNN):
    """CNN+CBAM hien tai (cnn_model_attention.CNN), them embed(). Khong doi kien truc."""

    def embed(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.cbam2 is not None:
            x = self.cbam2(x)
        if self.block3 is not None:
            x = self.cbam3(self.block3(x))
        return self.classifier[:-1](x)           # bo Linear cuoi -> (B, 2048)
