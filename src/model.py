import torch.nn as nn


def double_convolution(
        in_channels,
        out_channels,
        num_groups=8
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
    # for GroupNorm, ensure num_groups divides out_channels
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


class UNet(nn.Module): # TODO refine class description
    """
    This class defines the U-Net architecture.
    """

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.sslHead = SSLHead(1024)
        # define pooling layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # define double convolutions for encoder
        self.down_convolution_1 = double_convolution(1, 64)  # in encoder stage 1
        self.down_convolution_2 = double_convolution(64, 128)  # in encoder stage 2
        self.down_convolution_3 = double_convolution(128, 256)  # in encoder stage 3
        self.down_convolution_4 = double_convolution(256, 512)  # in encoder stage 4
        self.down_convolution_5 = double_convolution(512, 1024)  # in bottleneck

        # ----------------- DEFINE DECODER
        # in decoder stage 1
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_1 = double_convolution(1024, 512)

        # in decoder stage 2
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256)

        # in decoder stage 3
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)

        # in decoder stage 4
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)

        # ----------------- DEFINE OUTPUT
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x, is_seg):
        # ----------------- ENCODER
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)

        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)

        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)

        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)

        down_9 = self.down_convolution_5(down_8)

        # ----------------- DECODER
        if (is_seg):  # If the path is segmentation we go through decoder
            up_1 = self.up_transpose_1(down_9)
            x = self.up_convolution_1(torch.cat([down_7, up_1], 1))

            up_2 = self.up_transpose_2(x)
            x = self.up_convolution_2(torch.cat([down_5, up_2], 1))

            up_3 = self.up_transpose_3(x)
            x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

            up_4 = self.up_transpose_4(x)
            x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

            # ----------------- OUTPUT
            out = self.out(x)
            return out
        else:  # else we send images to sslHead after the encoder
            return self.sslHead(down_9)