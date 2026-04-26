import torch.nn as nn


def double_convolution(
        in_channels,
        out_channels,
        num_groups: int | None = None,
) -> nn.Sequential:
    """
    This function creates two consecutive convolution layers. GroupNorm is optional: just pass num_groups=None to
    disable it. GroupNorm is off by default for safety.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        num_groups (int | None): number of groups for GroupNorm, also used to deactivate GN if passed as None. Set to
            None by default.
    Returns:
        conv_op (nn.sequential): two convolution layers.
    """
    if num_groups is not None and num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")

    use_groupnorm = num_groups is not None
    if use_groupnorm:
        # For GN: ensure num_groups divides out_channels cleanly
        g = min(num_groups, out_channels)
        while out_channels % g != 0:
            g -= 1

    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_channels) if use_groupnorm else nn.Identity(),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(g, out_channels) if use_groupnorm else nn.Identity(),
        nn.ReLU(inplace=True)
    )
    return conv_op
