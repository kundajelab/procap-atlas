"""Attribution-friendly wrappers for profile models."""

from __future__ import annotations

import torch


class OrientationIndexWrapper(torch.nn.Module):
    """Return orientation index from a model's profile logits."""

    def __init__(self, model, center_logits: bool = True):
        super().__init__()
        self.model = model
        self.center_logits = center_logits
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()

    def forward(self, X, X_ctl=None, **kwargs):
        if X_ctl is None:
            logits = self.model(X, **kwargs)[0]
        else:
            logits = self.model(X, X_ctl, **kwargs)[0]

        flat = logits.flatten(start_dim=1)
        if self.center_logits:
            flat = flat - flat.mean(dim=-1, keepdim=True)

        probabilities = self.softmax(flat).reshape_as(logits)
        strand_mass = probabilities.sum(dim=-1)
        forward = strand_mass[:, 0:1]
        reverse = strand_mass[:, 1:2]
        return forward + self.relu(reverse - forward)
