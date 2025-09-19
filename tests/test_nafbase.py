import pytest  # noqa: F401
import torch

from extrasvls.models.nafbaseline import NAFBaseline


def test_nafbase_shape() -> None:
    sample = torch.randn((4, 3, 256, 256))
    nafb = NAFBaseline(3, 16, 3, [[4, 4, 4, 4], [4], [4, 4, 4, 4]], torch.tensor([256, 256]))
    out = nafb(sample)
    assert out.shape == (4, 3, 256, 256)
