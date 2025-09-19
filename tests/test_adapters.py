import pytest
import torch
from torch.nn.functional import one_hot

from extrasvls.adapters import (
    BaseAdapter,
    IdentityAdapter,
    InvertedGaussAdapter,
    RandomAdapter,
    UniformAdapter,
    apply_adapter,
    load_adapter,
)


def test_identity_adapter() -> None:
    adapter = IdentityAdapter()
    x = torch.ones((1, 1, 3, 3))
    out = adapter(x)
    assert (out == x).all()


def test_base_adapter() -> None:
    adapter = BaseAdapter()
    weight = adapter.weight.squeeze()
    true_values = [
        weight[0, 0] + 2 * weight[0, 1] + weight[1, 1],
        2 * weight[0, 0] + 3 * weight[0, 1] + weight[1, 1],
    ]

    x = torch.ones((1, 1, 3, 3))
    out = adapter(x).squeeze()
    assert torch.isclose(out[0, 0], true_values[0])
    assert torch.isclose(out[0, 1], true_values[1])
    assert torch.isclose(out[1, 1], torch.tensor(1.0))


def test_uniform_adapter() -> None:
    adapter = UniformAdapter()
    weight = adapter.weight.squeeze()
    true_values = [
        weight[0, 0] + 2 * weight[0, 1] + weight[1, 1],
        2 * weight[0, 0] + 3 * weight[0, 1] + weight[1, 1],
    ]

    x = torch.ones((1, 1, 3, 3))
    out = adapter(x).squeeze()
    assert torch.isclose(out[0, 0], true_values[0])
    assert torch.isclose(out[0, 1], true_values[1])
    assert torch.isclose(out[1, 1], torch.tensor(1.0))


def test_inverted_gauss_adapter() -> None:
    adapter = InvertedGaussAdapter()
    weight = adapter.weight.squeeze()
    true_values = [
        weight[0, 0] + 2 * weight[0, 1] + weight[1, 1],
        2 * weight[0, 0] + 3 * weight[0, 1] + weight[1, 1],
    ]

    x = torch.ones((1, 1, 3, 3))
    out = adapter(x).squeeze()
    assert torch.isclose(out[0, 0], true_values[0])
    assert torch.isclose(out[0, 1], true_values[1])
    assert torch.isclose(out[1, 1], torch.tensor(1.0))


def test_random_adapter() -> None:
    torch.manual_seed(42)  # type: ignore
    adapter = RandomAdapter()
    weight = adapter.weight.squeeze()
    true_values = [
        weight[1, 1] + weight[1, 2] + weight[2, 1] + weight[2, 2],
        weight[1, 0] + weight[1, 1] + weight[1, 2] + weight[2, 0] + weight[2, 1] + weight[2, 2],
    ]

    x = torch.ones((1, 1, 3, 3))
    out = adapter(x).squeeze()
    assert torch.isclose(out[0, 0], true_values[0]), "Corner"
    assert torch.isclose(out[0, 1], true_values[1]), "Edge"
    assert torch.isclose(out[1, 1], torch.tensor(1.0)), "Center"


def test_load_base_adapter() -> None:
    assert isinstance(load_adapter("base"), BaseAdapter)


def test_improper_adapter_name_load_raises_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        load_adapter("wrong_name")
    assert "wrong_name is not a supported adapter type." in str(excinfo.value)


def test_apply_adapter() -> None:
    torch.manual_seed(42)  # type: ignore
    expert_labels = torch.randn((4, 6, 3, 256, 256))
    expert_labels = one_hot(torch.argmax(expert_labels, dim=2)).movedim(-1, 2).float()  # pylint: disable=not-callable

    adapter = IdentityAdapter()
    adapter_labels = apply_adapter(expert_labels, adapter)
    assert torch.isclose(adapter_labels, expert_labels.mean(dim=1)).all()
