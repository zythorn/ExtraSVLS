from typing import Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def apply_adapter(labels: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """
    Labels of shape [batch_size, num_experts, num_classes, height, width].
    """
    print(f"Applying adapter: {labels.shape}")
    num_experts, num_classes = labels.shape[1], labels.shape[2]
    expert_labels: list[torch.Tensor] = []
    for expert in range(num_experts):
        svls = torch.stack([adapter(labels[:, expert, [cls]]) for cls in range(num_classes)], dim=2)
        expert_labels.append(svls)
    smooth_labels = torch.stack(expert_labels).mean(dim=0).squeeze(dim=1)
    return smooth_labels


class IdentityAdapter(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class UniformAdapter(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()  # type: ignore

        self.weight = torch.ones((1, 1, 3, 3))
        self.weight /= self.weight.sum()
        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)  # pylint: disable=not-callable


class RandomAdapter(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()  # type: ignore
        self.weight = nn.Parameter(torch.rand((1, 1, 3, 3)), requires_grad=False)
        self.weight /= self.weight.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.weight = nn.Parameter(torch.rand((1, 1, 3, 3)), requires_grad=False)
        # print(self.weight)
        return F.conv2d(x, self.weight, padding=1)  # pylint: disable=not-callable


class BaseAdapter(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()  # type: ignore

        gauss = [np.exp(-x / 2.0) / (2.0 * np.pi) for x in [1, 0, 1]]
        x, y = torch.tensor(gauss), torch.tensor(gauss)
        self.weight = x[:, torch.newaxis] @ y[torch.newaxis, :]

        border_sum = self.weight.sum() - self.weight[1, 1]
        self.weight[1, 1] = border_sum
        self.weight = (self.weight / (2.0 * border_sum)).reshape((1, 1, 3, 3)).float()
        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)  # pylint: disable=not-callable


class InvertedGaussAdapter(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()  # type: ignore

        gauss = [np.exp(-x / 2.0) / (2.0 * np.pi) for x in [0, 1, 0]]
        x, y = torch.tensor(gauss), torch.tensor(gauss)
        self.weight = x[:, torch.newaxis] @ y[torch.newaxis, :]

        # border_sum = self.weight.sum() - self.weight[1, 1]
        # self.weight[1, 1] = border_sum
        self.weight = (self.weight / self.weight.sum()).reshape((1, 1, 3, 3)).float()
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        print(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)  # pylint: disable=not-callable


def load_adapter(adapter_name: str) -> nn.Module:
    """
    Creates an instance of an adapter by a specified name.
    """
    adapter_dict: dict[str, Type[nn.Module]] = {
        "base": BaseAdapter,
        "identity": IdentityAdapter,
        "uniform": UniformAdapter,
        "random": RandomAdapter,
        "inverted": InvertedGaussAdapter,
    }
    if adapter_name not in adapter_dict.keys():
        raise ValueError(f"{adapter_name} is not a supported adapter type.")
    return adapter_dict[adapter_name]()


def test() -> None:
    """
    Run a basic adapter test.
    """
    label = torch.tensor(list(range(900))).reshape((2, 6, 3, 5, 5)).float()
    print(label.shape)
    adapter = InvertedGaussAdapter()
    svls = apply_adapter(label, adapter)
    print(svls.shape)
    print(svls[0, 0])


if __name__ == "__main__":
    test()
