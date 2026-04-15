import torch
import torch.nn as nn
import torchvision.transforms as T


def make_ssl_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: # TODO fix docstring
    """


    Args:
        x (torch.Tensor): input images (B, C, H, W)
    Returns:
        x_ssl (torch.Tensor): cropped and rotated images (B, C, h, w)
        y_ssl (torch.Tensor): rotation labels (B,) in {0,1,2,3}
    """
    B, C, H, W = x.shape
    x_ssl_list = []
    y_ssl_list = []

    # possibili rotazioni in multipli di 90°
    angles = [0, 90]
    ks = [0, 1]  # corrispondente k per torch.rot90

    for i in range(B):
        img = x[i]                              # (C,H,W)
        img = center_crop(img)                  # center crop

        # scegli random una delle 4 rotazioni
        idx = torch.randint(0, 2,(1,)).item()
        k = ks[idx]

        # rotazione di k*90° senza interpolazione
        img_rot = torch.rot90(img, k=k, dims=(1, 2))

        x_ssl_list.append(img_rot)
        y_ssl_list.append(idx)

    x_ssl = torch.stack(x_ssl_list, dim=0)          # (B,C,h,w)
    y_ssl = torch.tensor(y_ssl_list, dtype=torch.long, device=x.device)  # (B,)

    return x_ssl, y_ssl


class SSLHead(nn.Module): # TODO describe what this class does
    """

    """
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.conv = double_convolution(in_channels,32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, feat):
        x = self.conv(feat)
        x = self.pool(x).flatten(1)  # (B,32)
        return self.fc(x)            # (B,4)