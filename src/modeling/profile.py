"""Shared helpers for two-strand profile model outputs."""

from __future__ import annotations

import numpy as np
import torch


def _is_torch(value) -> bool:
    return isinstance(value, torch.Tensor)


def _flatten_profile(profile_logits):
    if profile_logits.ndim < 3:
        raise ValueError(
            "profile_logits must have shape (N, 2, L) or compatible trailing "
            f"profile dimensions; got {tuple(profile_logits.shape)}"
        )
    return profile_logits.reshape(profile_logits.shape[0], -1)


def profile_probabilities(profile_logits, center: bool = True):
    """Return joint profile probabilities over all strands and positions."""
    if _is_torch(profile_logits):
        flat = _flatten_profile(profile_logits)
        if center:
            flat = flat - flat.mean(dim=-1, keepdim=True)
        return torch.softmax(flat, dim=-1).reshape_as(profile_logits)

    logits = np.asarray(profile_logits, dtype=np.float64)
    flat = _flatten_profile(logits)
    if center:
        flat = flat - flat.mean(axis=-1, keepdims=True)
    shifted = flat - np.max(flat, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities.reshape(logits.shape)


def profile_log_probabilities(profile_logits, center: bool = True):
    """Return joint profile log-probabilities over strands and positions."""
    if _is_torch(profile_logits):
        flat = _flatten_profile(profile_logits)
        if center:
            flat = flat - flat.mean(dim=-1, keepdim=True)
        return torch.log_softmax(flat, dim=-1).reshape_as(profile_logits)

    probabilities = profile_probabilities(profile_logits, center=center)
    return np.log(np.clip(probabilities, 1e-300, None))


def count_scaled_profile(profile_logits, log_counts, center: bool = True):
    """Scale joint profile probabilities by predicted total counts."""
    probabilities = profile_probabilities(profile_logits, center=center)
    if _is_torch(probabilities):
        if isinstance(log_counts, torch.Tensor):
            log_counts_tensor = log_counts.to(
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
        else:
            log_counts_tensor = torch.as_tensor(
                log_counts,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
        counts = torch.exp(log_counts_tensor)
        counts = counts.reshape(probabilities.shape[0], -1)
        if counts.shape[1] != 1:
            raise ValueError(
                "log_counts must contain one count prediction per locus; "
                f"got shape {tuple(torch.as_tensor(log_counts).shape)}"
            )
        scale_shape = (probabilities.shape[0],) + (1,) * (probabilities.ndim - 1)
        return probabilities * counts.reshape(scale_shape)

    counts = np.asarray(log_counts, dtype=np.float64).reshape(probabilities.shape[0], -1)
    if counts.shape[1] != 1:
        raise ValueError(
            "log_counts must contain one count prediction per locus; "
            f"got shape {np.asarray(log_counts).shape}"
        )
    scale_shape = (probabilities.shape[0],) + (1,) * (probabilities.ndim - 1)
    return probabilities * np.exp(counts).reshape(scale_shape)


def strand_masses(profile_logits, center: bool = True):
    """Return forward and reverse strand masses from joint profile probabilities."""
    probabilities = profile_probabilities(profile_logits, center=center)
    return strand_masses_from_profile(probabilities)


def strand_masses_from_profile(profile):
    """Return forward and reverse strand masses from a two-strand profile."""
    if _is_torch(profile):
        if profile.ndim != 3 or profile.shape[1] != 2:
            raise ValueError(
                "profile must have shape (N, 2, L); "
                f"got {tuple(profile.shape)}"
            )
        return profile.sum(dim=-1)

    profile = np.asarray(profile)
    if profile.ndim != 3 or profile.shape[1] != 2:
        raise ValueError(
            "profile must have shape (N, 2, L); "
            f"got {tuple(profile.shape)}"
        )
    return profile.sum(axis=-1)


def orientation_index(
    profile,
    is_logit: bool = False,
    center: bool = True,
    eps: float = 1e-12,
):
    """Return max strand mass divided by total mass for profiles or logits."""
    if is_logit:
        profile = profile_probabilities(profile, center=center)

    masses = strand_masses_from_profile(profile)
    if _is_torch(masses):
        total = masses.sum(dim=1, keepdim=True)
        maximum = masses[:, 0:1] + torch.relu(masses[:, 1:2] - masses[:, 0:1])
        return maximum / total.clamp_min(eps)

    total = masses.sum(axis=1, keepdims=True)
    maximum = masses[:, 0:1] + np.maximum(masses[:, 1:2] - masses[:, 0:1], 0)
    return np.divide(
        maximum,
        np.maximum(total, eps),
        out=np.zeros_like(maximum, dtype=np.result_type(maximum, float)),
        where=total > 0,
    )
