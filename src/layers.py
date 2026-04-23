import torch.nn as nn


def double_convolution(
        in_channels,
        out_channels,
        num_groups,
) -> nn.Sequential:
    """
    This function creates two consecutive convolution layers.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        num_groups (int): number of groups for GroupNorm.
    Returns:
        conv_op (nn.sequential): two convolution layers.
    """
    # For GroupNorm, ensure num_groups divides out_channels
    g = min(num_groups, out_channels)
    while out_channels % g != 0:
        g -= 1

    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_channels),  # GroupNorm
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_channels),  # GroupNorm
        nn.ReLU(inplace=True)
    )
    return conv_op