"""Utilities for BPNet locus-level attribution diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from bpnetlite.bpnet import _ProfileLogitScaling
from tangermeme.deep_lift_shap import (
    dinucleotide_shuffle,
    hypothetical_attributions,
)
from tangermeme.predict import predict
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

ALPHABET = "ACGT"
CACHE_SCHEMA_VERSION = 4


class StrandProfileWrapper(torch.nn.Module):
    """Return one strand's profile score after joint two-strand softmax weighting."""

    def __init__(self, model, strand):
        super().__init__()
        if strand not in (0, 1):
            raise ValueError("strand must be 0 (plus) or 1 (minus)")
        self.model = model
        self.strand = strand
        self.scaling = _ProfileLogitScaling()

    def forward(self, X, X_ctl=None, **kwargs):
        if X_ctl is None:
            logits = self.model(X, **kwargs)[0]
        else:
            logits = self.model(X, X_ctl, **kwargs)[0]
        flattened = logits.flatten(start_dim=1)
        centered = flattened - flattened.mean(dim=-1, keepdim=True)
        weighted = self.scaling(centered).reshape_as(logits)
        return weighted[:, self.strand].sum(dim=-1, keepdim=True)


def per_reference_attributions(multipliers, genomic_input, references):
    """Project raw DeepLIFT multipliers into observed per-reference attributions."""
    multipliers = torch.as_tensor(multipliers)
    references = torch.as_tensor(references)
    if multipliers.ndim == 4 and multipliers.shape[0] == 1:
        multipliers = multipliers[0]
    if references.ndim == 4 and references.shape[0] == 1:
        references = references[0]
    if multipliers.shape != references.shape:
        raise ValueError("multipliers and references must have matching shapes")

    genomic_input = torch.as_tensor(genomic_input)
    if genomic_input.ndim == 3 and genomic_input.shape[0] == 1:
        genomic_input = genomic_input[0]
    if genomic_input.shape != multipliers.shape[1:]:
        raise ValueError("genomic_input must match one reference sequence")

    repeated_input = genomic_input.unsqueeze(0).expand_as(references)
    hypothetical = hypothetical_attributions(
        (multipliers,),
        (repeated_input,),
        (references,),
    )[0]
    return hypothetical * repeated_input


def reference_weight_schemes(probabilities, counts, temperature=1.0, window=20):
    """Calculate shared reference weights from fold-level profile predictions."""
    probabilities = as_numpy(probabilities).astype(float)
    counts = as_numpy(counts).astype(float)
    if probabilities.ndim == 3:
        probabilities = probabilities[None]
    if counts.ndim == 1:
        counts = counts[None]
    if (
        probabilities.ndim != 4
        or probabilities.shape[2] != 2
        or counts.shape != probabilities.shape[:2]
    ):
        raise ValueError(
            "probabilities must have shape (folds, references, 2, positions) "
            "and counts must have shape (folds, references)"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    folds, references = probabilities.shape[:2]
    maxima, _, _ = rolling_profile_maxima(
        probabilities.reshape(folds * references, 2, probabilities.shape[-1]),
        windows=(window,),
    )
    max_window_mass = maxima[window].reshape(folds, references).mean(axis=0)
    strand_mass = probabilities.sum(axis=-1)
    strand_imbalance = np.abs(strand_mass[..., 0] - strand_mass[..., 1]).mean(
        axis=0
    )
    log_counts = np.log1p(counts).mean(axis=0)

    metrics = np.stack([max_window_mass, strand_imbalance, log_counts])
    penalties = np.zeros_like(metrics)
    for index, values in enumerate(metrics):
        median = np.median(values)
        deviations = np.abs(values - median)
        scale = 1.4826 * np.median(deviations)
        if scale <= 1e-12:
            scale = deviations.mean()
        if scale > 1e-12:
            penalties[index] = np.maximum(values - median, 0) / scale

    contamination = np.stack(
        [
            np.zeros(references),
            penalties[:2].sum(axis=0),
            penalties.sum(axis=0),
        ]
    )
    shifted = -contamination / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    return {
        "metric_names": np.asarray(
            ["max_20bp_probability", "strand_imbalance", "log1p_counts"]
        ),
        "metrics": metrics,
        "contamination": contamination,
        "weights": weights,
    }


def weighted_strand_attributions(attributions, weights):
    """Combine per-reference strand attributions with shared reference weights."""
    attributions = as_numpy(attributions)
    weights = as_numpy(weights)
    if attributions.ndim < 3 or attributions.shape[-3] != len(weights):
        raise ValueError("reference axis must be third from last")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("weights must be nonnegative and sum to one")
    return np.tensordot(
        attributions,
        weights,
        axes=([-3], [0]),
    )


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def channel_first(sequence):
    sequence = as_numpy(sequence)
    if sequence.ndim == 3:
        sequence = sequence[0]
    if sequence.shape[0] == 4:
        return sequence
    if sequence.shape[-1] == 4:
        return sequence.T
    raise ValueError(f"Expected a four-channel sequence, got {sequence.shape}")


def dinucleotide_frequencies(sequence):
    """Return normalized genomic dinucleotide frequencies in ACGT order."""
    sequence = channel_first(sequence)
    bases = sequence.argmax(axis=0)
    valid = sequence.sum(axis=0) > 0
    counts = np.zeros((4, 4), dtype=float)
    for left, right, keep in zip(bases[:-1], bases[1:], valid[:-1] & valid[1:]):
        if keep:
            counts[left, right] += 1
    total = counts.sum()
    return counts / total if total else counts


def reference_banks(X, seeds, n_references):
    """Generate reproducible DeepLIFT dinucleotide-shuffled reference banks."""
    return {
        int(seed): dinucleotide_shuffle(
            X.cpu(), n=n_references, random_state=int(seed)
        )[0]
        for seed in seeds
    }


def low_activity_references(
    genomic_input,
    selector_models,
    n_candidates,
    n_references,
    random_state,
    batch_size,
    device,
    min_hamming_fraction=0.1,
):
    """Select diverse shuffles with low activity across held-out selector models.

    ``selector_models`` may be a generator that loads one model at a time. For
    attribution of fold k, it should yield only the other folds so the explained
    model does not select its own baseline. Candidates are ranked by their worst
    percentile across selector folds and three metrics: predicted counts,
    maximum 20 bp count-scaled signal, and profile concentration.
    """
    if genomic_input.shape[0] != 1:
        raise ValueError("genomic_input must contain exactly one sequence")
    if not 0 <= min_hamming_fraction <= 1:
        raise ValueError("min_hamming_fraction must be between zero and one")
    if n_references < 1 or n_candidates < n_references:
        raise ValueError("n_candidates must be at least n_references")

    candidates = dinucleotide_shuffle(
        genomic_input.cpu(),
        n=n_candidates,
        random_state=int(random_state),
    )[0]
    fold_counts = []
    fold_max_20bp = []
    fold_profile_score = []

    for model in selector_models:
        profile_logits, log_counts = predict(
            model=model,
            X=candidates,
            batch_size=batch_size,
            device=device,
        )
        logits = torch.as_tensor(profile_logits).reshape(n_candidates, -1)
        centered = logits - logits.mean(dim=1, keepdim=True)
        probabilities = torch.softmax(centered, dim=1)
        counts = torch.exp(
            torch.as_tensor(log_counts).reshape(n_candidates, -1)
        ).sum(dim=1)
        count_scaled = (probabilities * counts[:, None]).reshape(
            torch.as_tensor(profile_logits).shape
        )
        maxima, _, _ = rolling_profile_maxima(
            as_numpy(count_scaled), windows=(20,)
        )
        fold_counts.append(as_numpy(counts))
        fold_max_20bp.append(maxima[20])
        fold_profile_score.append(
            as_numpy((centered * probabilities).sum(dim=1))
        )

    if not fold_counts:
        raise ValueError("selector_models must contain at least one model")

    metrics = np.stack(
        [
            np.asarray(fold_counts),
            np.asarray(fold_max_20bp),
            np.asarray(fold_profile_score),
        ],
        axis=1,
    )
    percentile_ranks = np.empty_like(metrics, dtype=float)
    for fold in range(metrics.shape[0]):
        for metric in range(metrics.shape[1]):
            order = np.argsort(metrics[fold, metric], kind="stable")
            percentile_ranks[fold, metric, order] = (
                np.arange(n_candidates) + 1
            ) / n_candidates
    selection_score = percentile_ranks.max(axis=(0, 1))
    ranked_indices = np.argsort(selection_score, kind="stable")

    bases = candidates.argmax(dim=1).numpy()
    selected_indices = []
    for index in ranked_indices:
        if all(
            np.mean(bases[index] != bases[chosen]) >= min_hamming_fraction
            for chosen in selected_indices
        ):
            selected_indices.append(int(index))
            if len(selected_indices) == n_references:
                break
    if len(selected_indices) < n_references:
        selected = set(selected_indices)
        selected_indices.extend(
            int(index)
            for index in ranked_indices
            if int(index) not in selected
        )
        selected_indices = selected_indices[:n_references]

    selected_indices = np.asarray(selected_indices, dtype=int)
    return {
        "references": candidates[selected_indices],
        "selected_indices": selected_indices,
        "selection_score": selection_score,
        "fold_counts": np.asarray(fold_counts),
        "fold_max_20bp": np.asarray(fold_max_20bp),
        "fold_profile_score": np.asarray(fold_profile_score),
    }


def markov_parameters(sequence):
    """Estimate initial and first-order transition probabilities."""
    sequence = channel_first(sequence)
    mono = sequence.sum(axis=1).astype(float)
    initial = mono / mono.sum()
    dinuc = dinucleotide_frequencies(sequence)
    transition = dinuc.copy()
    for i in range(4):
        row_sum = transition[i].sum()
        transition[i] = transition[i] / row_sum if row_sum else initial
    return initial, transition


def sample_markov_sequences(sequence, length, n, random_state):
    """Sample one-hot sequences from the input's first-order Markov model."""
    initial, transition = markov_parameters(sequence)
    rng = np.random.default_rng(random_state)
    bases = np.empty((n, length), dtype=np.int64)
    bases[:, 0] = rng.choice(4, size=n, p=initial)
    for position in range(1, length):
        for i in range(n):
            bases[i, position] = rng.choice(4, p=transition[bases[i, position - 1]])
    sampled = np.zeros((n, 4, length), dtype=np.int8)
    rows = np.arange(n)[:, None]
    positions = np.arange(length)[None, :]
    sampled[rows, bases, positions] = 1
    return torch.from_numpy(sampled)


def gradient_x_input(model, X, device):
    """Calculate observed gradient-times-input attribution."""
    model = model.to(device)
    model.eval()
    x = X.detach().clone().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(x)
    output.sum().backward()
    attribution = x.grad * x
    return attribution.detach().cpu()


def point_ism(model, X, start, end, batch_size, device):
    """Calculate observed single-nucleotide ISM over a sequence interval."""
    return saturation_mutagenesis(
        model=model,
        X=X,
        start=start,
        end=end,
        batch_size=batch_size,
        hypothetical=False,
        device=device,
    ).cpu()


def window_ism(
    model,
    X,
    start,
    end,
    window,
    stride,
    n_replacements,
    random_state,
    batch_size,
    device,
):
    """Measure original-minus-perturbed activity for sliding Markov windows."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    positions = np.arange(start, end - window + 1, stride, dtype=int)
    original = as_numpy(
        predict(model, X, batch_size=1, device=device)
    ).reshape(X.shape[0], -1)
    scores = []
    for i, position in enumerate(positions):
        replacements = sample_markov_sequences(
            X, window, n_replacements, random_state + i
        )
        perturbed = X.repeat(n_replacements, 1, 1)
        perturbed[:, :, position : position + window] = replacements
        activity = as_numpy(
            predict(model, perturbed, batch_size=batch_size, device=device)
        ).reshape(n_replacements, -1)
        scores.append((original[0] - activity.mean(axis=0)).mean())
    return positions, np.asarray(scores)


def rolling_profile_maxima(profiles, windows=(1, 5, 20)):
    """Return strongest strand-specific rolling sums and peak coordinates."""
    profiles = as_numpy(profiles)
    if profiles.ndim != 3 or profiles.shape[1] != 2:
        raise ValueError("profiles must have shape (sequences, 2, positions)")
    maxima = {}
    peak_positions = {}
    peak_strands = {}
    cumulative = np.pad(
        np.cumsum(profiles, axis=-1), ((0, 0), (0, 0), (1, 0))
    )
    for window in windows:
        if window < 1 or window > profiles.shape[-1]:
            raise ValueError("window must fit inside the profile")
        sums = cumulative[..., window:] - cumulative[..., :-window]
        flattened = sums.reshape(len(profiles), -1)
        indices = flattened.argmax(axis=1)
        maxima[window] = flattened[np.arange(len(profiles)), indices]
        peak_strands[window] = indices // sums.shape[-1]
        peak_positions[window] = indices % sums.shape[-1]
    return maxima, peak_positions, peak_strands


def profile_summaries(probabilities, count_scaled_profiles, windows=(1, 5, 20)):
    """Summarize profile concentration and strongest local count-scale signal."""
    probabilities = as_numpy(probabilities)
    count_scaled_profiles = as_numpy(count_scaled_profiles)
    maxima, positions, strands = rolling_profile_maxima(
        count_scaled_profiles, windows=windows
    )
    flat = probabilities.reshape(len(probabilities), -1)
    entropy = -(flat * np.log(flat.clip(min=1e-12))).sum(axis=1)
    summaries = {
        "profile_entropy": entropy,
        "effective_width": np.exp(entropy),
    }
    for window in windows:
        summaries[f"max_{window}bp"] = maxima[window]
        summaries[f"peak_{window}bp_position"] = positions[window]
        summaries[f"peak_{window}bp_strand"] = strands[window]
    return summaries


def fold_consensus(peak_positions, peak_strands, peak_activity):
    """Summarize agreement across the leading fold axis."""
    peak_positions = as_numpy(peak_positions)
    peak_strands = as_numpy(peak_strands)
    peak_activity = as_numpy(peak_activity)
    if not (
        peak_positions.shape == peak_strands.shape == peak_activity.shape
        and peak_positions.ndim >= 1
    ):
        raise ValueError("fold consensus inputs must have matching shapes")
    strand_fraction = np.maximum(
        (peak_strands == 0).mean(axis=0), (peak_strands == 1).mean(axis=0)
    )
    mean_activity = peak_activity.mean(axis=0)
    return {
        "peak_position_sd": peak_positions.std(axis=0),
        "peak_strand_agreement": strand_fraction,
        "peak_activity_mean": mean_activity,
        "peak_activity_cv": peak_activity.std(axis=0)
        / np.maximum(mean_activity, 1e-12),
    }


def predicted_activity(model, genomic_input, references, batch_size, device):
    """Predict complete profiles and summaries for genomic and reference inputs."""
    sequences = torch.cat([genomic_input.cpu(), references.cpu()])
    profile_logits, log_counts = predict(
        model=model,
        X=sequences,
        batch_size=batch_size,
        device=device,
    )
    profile_logits = torch.as_tensor(profile_logits)
    original_shape = profile_logits.shape
    flattened = profile_logits.reshape(len(sequences), -1)
    centered = flattened - flattened.mean(dim=-1, keepdim=True)
    log_probabilities = torch.log_softmax(centered, dim=-1)
    probabilities = log_probabilities.exp()
    counts = torch.exp(torch.as_tensor(log_counts).reshape(len(sequences), -1)).sum(
        dim=-1
    )
    count_scaled = probabilities * counts[:, None]
    genomic_profile = probabilities[:1]
    midpoint = 0.5 * (probabilities[1:] + genomic_profile)
    jsd = 0.5 * (
        (
            probabilities[1:]
            * (log_probabilities[1:] - torch.log(midpoint.clamp_min(1e-12)))
        ).sum(dim=-1)
        + (
            genomic_profile
            * (log_probabilities[:1] - torch.log(midpoint.clamp_min(1e-12)))
        ).sum(dim=-1)
    )
    profile_score = (centered * torch.softmax(centered, dim=-1)).sum(dim=-1)
    centered = centered.reshape(original_shape)
    probabilities = probabilities.reshape(original_shape)
    count_scaled = count_scaled.reshape(original_shape)
    summaries = profile_summaries(
        as_numpy(probabilities), as_numpy(count_scaled), windows=(1, 5, 20)
    )
    return {
        "genomic_centered_logits": as_numpy(centered[:1]),
        "genomic_probabilities": as_numpy(probabilities[:1]),
        "genomic_count_scaled": as_numpy(count_scaled[:1]),
        "genomic_profile_score": float(profile_score[0]),
        "genomic_counts": float(counts[0]),
        "reference_centered_logits": as_numpy(centered[1:]),
        "reference_probabilities": as_numpy(probabilities[1:]),
        "reference_count_scaled": as_numpy(count_scaled[1:]),
        "reference_profile_score": as_numpy(profile_score[1:]),
        "reference_counts": as_numpy(counts[1:]),
        "reference_profile_jsd": as_numpy(jsd),
        "reference_summaries": {
            key: value[1:] for key, value in summaries.items()
        },
        "genomic_summaries": {
            key: value[:1] for key, value in summaries.items()
        },
    }


def reverse_complement_matrix(matrix):
    return as_numpy(matrix)[..., [3, 2, 1, 0], ::-1]


def reverse_complement_tracks(plus, minus):
    return np.abs(as_numpy(minus)[::-1]), -np.abs(as_numpy(plus)[::-1])


def reverse_complement_peak_coordinates(positions, strands, length, window):
    positions = as_numpy(positions)
    strands = as_numpy(strands)
    return length - window - positions, 1 - strands


def genomic_offsets(center, interval_start, interval_end, input_length):
    input_start = center - input_length // 2
    start = interval_start - input_start
    end = interval_end - input_start
    if start < 0 or end > input_length or end <= start:
        raise ValueError("Requested interval is outside the model input window")
    return start, end


def diagnostic_cache_path(cache_dir, name, parameters):
    payload = json.dumps(parameters, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return Path(cache_dir) / f"{name}.{digest}.npz"
