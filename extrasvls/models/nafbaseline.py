import torch
import torch.nn as nn

# import torch.nn.functional as F


class ChannelAttention(nn.Module):  # type: ignore[misc]
    """
    Block for Channel Attention, also known as Squeeze-and-Excitation block.
    """

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()  # type: ignore

        self.hid_channels = in_channels // reduction

        self.avg = nn.AdaptiveAvgPool2d(output_size=1)

        self.attetion = nn.Sequential(
            *[
                nn.Conv2d(in_channels, self.hid_channels, kernel_size=1),  # type: ignore
                nn.ReLU(),
                nn.Conv2d(self.hid_channels, in_channels, kernel_size=1),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies channel attention to input.
        """
        stats = self.avg(x)
        attention_scores = self.attetion(stats)
        return x * attention_scores


class BaselineBlock(nn.Module):  # type: ignore[misc]
    """
    NAFBaseline network.
    """

    def __init__(self, in_channels: int, hid_channels: int, im_shape: torch.Tensor):
        super().__init__()  # type: ignore

        # Block 1
        normalized_shape: list[int] = [
            in_channels,
            int(im_shape[0].item()),
            int(im_shape[1].item()),
        ]
        self.ln1 = nn.LayerNorm(normalized_shape)

        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1)

        self.gelu = nn.GELU()
        self.ca = ChannelAttention(hid_channels)

        self.conv3 = nn.Conv2d(hid_channels, in_channels, kernel_size=1)

        # Block 2
        self.ln2 = nn.LayerNorm(normalized_shape)

        self.conv4 = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(hid_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies NAFB block to input.
        """
        y1 = self.conv2(self.conv1(self.ln1(x)))
        y1 = self.conv3(self.ca(self.gelu(y1)))
        x = x + y1

        y2 = self.conv4(self.ln2(x))
        y2 = self.conv5(self.gelu(y2))
        x = x + y2

        return x


class Downconv(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int):
        super().__init__()  # type: ignore

        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upconv(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int):
        super().__init__()  # type: ignore

        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class NAFBaseline(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        n_channels: int,
        hid_dim: int,
        n_classes: int,
        blocks_num: list[list[int]],
        im_shape: torch.Tensor,
    ):
        super().__init__()  # type: ignore

        self.in_conv = nn.Conv2d(n_channels, hid_dim, kernel_size=3, padding=1)
        current_dim = hid_dim
        current_shape = im_shape

        self.encoders = nn.ModuleList()
        self.downconvs = nn.ModuleList()

        for num in blocks_num[0]:
            self.encoders.append(
                nn.Sequential(
                    *[
                        BaselineBlock(current_dim, current_dim * 2, current_shape)
                        for _ in range(num)
                    ]
                )
            )
            self.downconvs.append(Downconv(current_dim))
            current_dim *= 2
            current_shape //= 2

        self.bottleneck = nn.Sequential(
            *[
                BaselineBlock(current_dim, current_dim * 2, current_shape)
                for _ in range(blocks_num[1][0])
            ]
        )

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for num in blocks_num[2]:
            self.upconvs.append(Upconv(current_dim))
            current_dim //= 2
            current_shape *= 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        BaselineBlock(current_dim, current_dim * 2, current_shape)
                        for _ in range(num)
                    ]
                )
            )

        self.out_conv = nn.Conv2d(current_dim, n_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outs: list[torch.Tensor] = []

        x = self.in_conv(x)

        for encoder, downconv in zip(self.encoders, self.downconvs, strict=False):
            x = encoder(x)
            encoder_outs.append(x)
            x = downconv(x)

        x = self.bottleneck(x)

        for decoder, upconv, encoder_out in zip(
            self.decoders, self.upconvs, encoder_outs[::-1], strict=False
        ):
            # print(x.shape, encoder_out.shape)
            x = upconv(x)
            x += encoder_out
            x = decoder(x)

        x = self.out_conv(x)
        return x
