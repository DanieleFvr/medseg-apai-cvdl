import torch
import torch.nn as nn

from src.layers import double_convolution
from src.ssl import SSLHead


class TeacherUNet(nn.Module):  # TODO refine class description
    """
    This class defines the student model architecture.
    """

    def __init__(
            self,
            num_groups,
    ):
        super(TeacherUNet, self).__init__()
        self.num_classes = 1
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)  # Define pooling layer
        self.sslHead = SSLHead(  # TODO rename sslHead --> ssl_head
            in_channels=1024,  # TODO hardcoded like this is not right
            num_groups=num_groups,
        )

        # Define double convolutions for encoder
        self.down_convolution_1 = double_convolution(1, 64, num_groups)  # In encoder stage 1
        self.down_convolution_2 = double_convolution(64, 128, num_groups)  # In encoder stage 2
        self.down_convolution_3 = double_convolution(128, 256, num_groups)  # In encoder stage 3
        self.down_convolution_4 = double_convolution(256, 512, num_groups)  # In encoder stage 4
        self.down_convolution_5 = double_convolution(512, 1024, num_groups)  # In bottleneck

        # ----------------- DEFINE DECODER

        # In decoder stage 1
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2,
        )
        self.up_convolution_1 = double_convolution(1024, 512, num_groups)

        # In decoder stage 2
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2,
        )
        self.up_convolution_2 = double_convolution(512, 256, num_groups)

        # In decoder stage 3
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
        )
        self.up_convolution_3 = double_convolution(256, 128, num_groups)

        # In decoder stage 4
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
        )
        self.up_convolution_4 = double_convolution(128, 64, num_groups)

        # ----------------- DEFINE OUTPUT

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=1,
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

        if is_seg:  # If the path is segmentation we go through decoder
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
        else:  # Else we send images to sslHead after the encoder
            return self.sslHead(down_9)


class StudentUNet(nn.Module):
    def __init__(
            self,
            num_groups,
    ):
        super().__init__()
        self.num_classes = 1

        # define encoder convolutions
        self.down_convolution_1 = double_convolution(1, 32, num_groups)
        self.down_convolution_2 = double_convolution(32, 64, num_groups)
        self.down_convolution_3 = double_convolution(64, 128, num_groups)
        self.down_convolution_4 = double_convolution(128, 256, num_groups)

        # define bottleneck convolutions
        self.down_convolution_bot = double_convolution(256, 512, num_groups)

        # define encoder downsampling layers
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # define
        self.up_transpose_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_transpose_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_transpose_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_transpose_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # define decoder convolutions
        self.up_convolution_1 = double_convolution(512, 256, num_groups)
        self.up_convolution_2 = double_convolution(256, 128, num_groups)
        self.up_convolution_3 = double_convolution(128, 64, num_groups)
        self.up_convolution_4 = double_convolution(64, 32, num_groups)

        # define output
        self.out = nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x):  # x = (B, 1, 512, 512)
        # encoder
        e1 = self.down_convolution_1(x)  # (B, 32, 512, 512)
        p1 = self.max_pool2d(e1)  # (B, 32, 256, 256)

        e2 = self.down_convolution_2(p1)  # (B, 64, 256, 256)
        p2 = self.max_pool2d(e2)  # (B, 64, 128, 128)

        e3 = self.down_convolution_3(p2)  # (B, 128, 128, 128)
        p3 = self.max_pool2d(e3)  # (B, 128, 64, 64)

        e4 = self.down_convolution_4(p3)  # (B, 256, 64, 64)
        p4 = self.max_pool2d(e4)  # (B, 256, 32, 32)

        # bottleneck
        bottleneck = self.down_convolution_bot(p4)  # (B, 512, 32, 32)

        # decoder
        t1 = self.up_transpose_1(bottleneck)  # (B, 256, 64, 64)
        c1 = torch.cat([e4, t1], dim=1)  # (B, 512, 64, 64)
        d1 = self.up_convolution_1(c1)  # (B, 256, 64, 64)

        t2 = self.up_transpose_2(d1)  # (B, 128, 128, 128)
        c2 = torch.cat([e3, t2], dim=1)  # (B, 256, 128, 128)
        d2 = self.up_convolution_2(c2)  # (B, 128, 128, 128)

        t3 = self.up_transpose_3(d2)  # (B, 64, 256, 256)
        c3 = torch.cat([e2, t3], dim=1)  # (B, 128, 256, 256)
        d3 = self.up_convolution_3(c3)  # (B, 64, 256, 256)

        t4 = self.up_transpose_4(d3)  # (B, 32, 512, 512)
        c4 = torch.cat([e1, t4], dim=1)  # (B, 64, 512, 512)
        d4 = self.up_convolution_4(c4)  # (B, 32, 512, 512)

        # output
        return self.out(d4)  # (B, 1, 512, 512)
