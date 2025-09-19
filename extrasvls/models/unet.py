import torch
import torch.nn as nn


class ConvBlock(nn.Module):  # type: ignore[misc]
    """
    Basic convolutional block:

        [Conv2d >> BatchNorm2d >> ReLU] x 2

    Args:
        in_channels (int): number of input channels.
        hid_channels (int): number of channels in a hidden layer.
        out_channels (int): number of output channels.
        kernel_size (int or tuple): size of the convolving kernel.
        padding (int, tuple or str): padding added to all four sides of the input.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()  # type: ignore

        self.conv_block = nn.Sequential(
            *[
                nn.Conv2d(  # type: ignore
                    in_channels=in_channels,
                    out_channels=hid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm2d(hid_channels),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class Upsampling(nn.Module):  # type: ignore[misc]
    """
    Upsampling block with a scale factor of 2:

        Upsample >> Conv2d

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()  # type: ignore

        self.upsample = nn.Sequential(
            *[
                nn.Upsample(scale_factor=2),  # type: ignore
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class UNet_Encoder(nn.Module):  # type: ignore[misc]
    """
    Encoder part of the UNet:

        [ConvBlock >> MaxPool2d] x 4 >> ConvBlock.

    Reduces image linear size in half for each MaxPool.
    Gradually increases the number of channels up until 1024.

    Args:
        in_channels (int): number of input channels.
    """

    def __init__(self, init_channels: int):
        super().__init__()  # type: ignore

        self.conv0 = ConvBlock(in_channels=init_channels, hid_channels=64, out_channels=64)
        self.conv1 = ConvBlock(in_channels=64, hid_channels=128, out_channels=128)
        self.conv2 = ConvBlock(in_channels=128, hid_channels=256, out_channels=256)
        self.conv3 = ConvBlock(in_channels=256, hid_channels=512, out_channels=512)
        self.conv4 = ConvBlock(in_channels=512, hid_channels=1024, out_channels=1024)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        e0 = self.conv0(x)
        e1 = self.conv1(self.pooling(e0))
        e2 = self.conv2(self.pooling(e1))
        e3 = self.conv3(self.pooling(e2))
        e4 = self.conv4(self.pooling(e3))

        encoder_outputs = [e0, e1, e2, e3]
        return e4, encoder_outputs


class UNet_Decoder(nn.Module):  # type: ignore[misc]
    """
    Decoder part of the UNet:

        [Upsampling >> ConvBlock] x 4

    After each Upsampling, the output is concatenated with an output of an encoder
    layer of the same depth. For more info, see https://arxiv.org/abs/1505.04597.

    Doubles image linear size for each Upsampling.
    Gradually decreases the number of channels up until the number of classes.

    Args:
        num_classes (int): number of classes to predict.
    """

    def __init__(self, num_classes: int):
        super().__init__()  # type: ignore

        self.up0 = Upsampling(in_channels=1024, out_channels=512)
        self.up1 = Upsampling(in_channels=512, out_channels=256)
        self.up2 = Upsampling(in_channels=256, out_channels=128)
        self.up3 = Upsampling(in_channels=128, out_channels=64)

        self.deconv0 = ConvBlock(in_channels=1024, hid_channels=512, out_channels=512)
        self.deconv1 = ConvBlock(in_channels=512, hid_channels=256, out_channels=256)
        self.deconv2 = ConvBlock(in_channels=256, hid_channels=128, out_channels=128)
        self.deconv3 = ConvBlock(in_channels=128, hid_channels=64, out_channels=64)

        self.final = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, encoder_outputs: list[torch.Tensor]) -> torch.Tensor:
        d0 = self.up0(x)
        d0 = torch.cat([encoder_outputs[3], d0], dim=1)
        d0 = self.deconv0(d0)

        d1 = self.up1(d0)
        d1 = torch.cat([encoder_outputs[2], d1], dim=1)
        d1 = self.deconv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([encoder_outputs[1], d2], dim=1)
        d2 = self.deconv2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([encoder_outputs[0], d3], dim=1)
        d3 = self.deconv3(d3)

        return self.final(d3)


class UNet(nn.Module):  # type: ignore[misc]
    """
    UNet architecture consisting of a UNet encoder and UNet decode with skip-connections.

    Args:
        in_channels (int): number of input channels.
        num_classes (int): number of classes to predict.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()  # type: ignore

        self.encoder = UNet_Encoder(init_channels=n_channels)
        self.decoder = UNet_Decoder(num_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, encoder_outputs = self.encoder(x)
        return self.decoder(x, encoder_outputs)
