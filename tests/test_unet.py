import pytest  # noqa: F401
import torch

from extrasvls.models.unet import UNet


def test_unet_shape() -> None:
    sample = torch.randn((4, 3, 256, 256))
    unet = UNet()
    out = unet(sample)
    assert out.shape == (4, 1, 256, 256)
