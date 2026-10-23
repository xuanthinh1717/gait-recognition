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
    """
    3 khoi Conv + BatchNorm + CBAM(64, reduction=4).
    Dung AdaptiveAvgPool2d((8, 2)) thay vi GAP(1x1) de giu cau truc doc (dau->chan)
    va trai-phai cua dang di truoc khi chieu sang vector out_dim.
    """

    def __init__(self, out_dim=256):
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
        # Giu 8 dai ngang (dau -> chan) x 2 nua (trai - phai) -> 64 * 8 * 2 = 1024
        self.pool = nn.AdaptiveAvgPool2d((8, 2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 2, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):                        # (N, 1, S, S)
        x = self.pool(self.conv(x)).flatten(1)   # (N, 1024)
        return self.fc(x)                        # (N, out_dim)


class CNNLSTM(nn.Module):
    """
    temporal="lstm": CNN tung frame -> LSTM -> gop theo thoi gian
    temporal="mean": CNN tung frame -> gop theo thoi gian (khong LSTM)
    pool: cach gop thoi gian khi lay embedding / logits
        "mean": trung binh dau ra qua cac buoc (mac dinh, hanh vi cu)
        "last": chi lay buoc cuoi (chi co nghia voi LSTM)
    """

    def __init__(self, num_classes, temporal="lstm", feat_dim=256, dropout=0.3, pool="mean"):
        super().__init__()
        assert temporal in ("lstm", "mean")
        assert pool in ("mean", "last")
        if pool == "last" and temporal != "lstm":
            raise ValueError("pool='last' chi dung cho temporal='lstm'")
        self.temporal = temporal
        self.pool = pool
        self.encoder = FrameEncoder(feat_dim)
        if temporal == "lstm":
            # hidden = feat_dim de B va E co cung so chieu embedding (256)
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        # BNNeck giup chuan hoa phan pho vector dac trung cho khoang cach cosine
        self.bn = nn.BatchNorm1d(feat_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def step_features(self, x):                  # x: (B, T, 1, S, S) -> (B, T, feat_dim)
        B, T = x.shape[:2]
        f = self.encoder(x.flatten(0, 1)).view(B, T, -1)
        if self.temporal == "lstm":
            f, _ = self.lstm(f)
        return f

    def embed(self, x):                          # (B, feat_dim)
        f = self.step_features(x)
        pooled = f[:, -1] if self.pool == "last" else f.mean(dim=1)
        return self.bn(pooled)

    def forward(self, x, return_embed=False):    # logits (B, C) hoac (logits, emb)
        emb = self.embed(x)
        logits = self.head(self.drop(emb))
        return (logits, emb) if return_embed else logits

    def forward_steps(self, x, return_embed=False):  # logits tai MOI buoc thoi gian (B, T, C)
        f = self.step_features(x)
        B, T, D = f.shape
        f_bn = self.bn(f.reshape(B * T, D)).view(B, T, D)
        steps = self.head(self.drop(f_bn))
        if return_embed:
            pooled = f[:, -1] if self.pool == "last" else f.mean(dim=1)
            return steps, self.bn(pooled)
        return steps


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

    def forward(self, x, return_embed=False):
        emb = self.embed(x)
        logits = self.classifier[-1](emb)
        return (logits, emb) if return_embed else logits
