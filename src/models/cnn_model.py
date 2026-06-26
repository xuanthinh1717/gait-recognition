import torch.nn as nn


class CNN(nn.Module):

    def __init__(
        self,
        num_classes,
        dropout=0.3
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        layers = [
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(512, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

