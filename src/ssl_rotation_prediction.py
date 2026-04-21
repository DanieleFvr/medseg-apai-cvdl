import torch
import torch.nn as nn

from layers import double_convolution


def make_ssl_batch(
        x: torch.Tensor,
        center_crop  # TODO specify type
) -> tuple[torch.Tensor, torch.Tensor]:  # TODO complete docstring
    """
    This function

    Args:
        x (torch.Tensor): input images (B, C, H, W).
        center_crop (torch.Tensor): center crop images (B, C, H, W).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: x_ssl and y_ssl (x_ssl is cropped and rotated images (B, C, h, w), while
        y_ssl is rotation labels (B,)).
    """
    B, C, H, W = x.shape # TODO variables should be lowercase
    x_ssl_list = []
    y_ssl_list = []
    ks = [0, 1]  # Corresponding k for torch.rot90

    for i in range(B):
        img = x[i]  # (C,H,W)
        img = center_crop(img)  # center crop

        # Choose one out of 2 rotations, randomly  # TODO comments and variable names account for 4 rotations, expected 2
        idx = torch.randint(0, 2, (1,)).item()
        k = ks[idx]

        # Rotation of k*90 degrees without interpolation
        img_rot = torch.rot90(img, k=k, dims=(1, 2))

        x_ssl_list.append(img_rot)
        y_ssl_list.append(idx)

    x_ssl = torch.stack(x_ssl_list, dim=0)  # (B,C,h,w)
    y_ssl = torch.tensor(y_ssl_list, dtype=torch.long, device=x.device)  # (B,)  # TODO add device to args

    return x_ssl, y_ssl


class SSLHead(nn.Module):  # TODO describe what this class does
    """

    """

    def __init__(
            self,
            in_channels,
            num_groups,  # TODO import from config when called
            num_classes=2
    ):
        super().__init__()
        self.conv = double_convolution(in_channels, 32, num_groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, feat):
        x = self.conv(feat)
        x = self.pool(x).flatten(1)  # (B,32)
        return self.fc(x)  # (B,num_classes)
