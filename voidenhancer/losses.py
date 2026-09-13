"""Losses used by the first VoidEnhancer training stage."""

import torch
from torch import nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1-like reconstruction loss that is stable for image restoration."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.eps ** 2))


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare simple image gradients to encourage clean edges."""
    px = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    py = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)
