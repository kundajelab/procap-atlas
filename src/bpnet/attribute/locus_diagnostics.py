"""Utilities for BPNet locus-level attribution diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tangermeme.deep_lift_shap import hypothetical_attributions
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.predict import predict
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

ALPHABET = "ACGT"
CACHE_SCHEMA_VERSION = 4


class StrandProfileWrapper(nn.Module):
    """Score one profile strand after a joint two-strand softmax."""

    def __init__(self, model, strand):
        super().__init__()
        self.model = model
        self.strand = strand

    def forward(self, X):
        profile_logits, _ = self.model(X)
        flat = profile_logits.reshape(profile_logits.shape[0], -1)
        centered = flat - flat.mean(dim=1, keepdim=True)
        probabilities = torch.softmax(centered, dim=1).reshape_as(profile_logits)
        centered = centered.reshape_as(profile_logits)
        score = (probabilities[:, self.strand] * centered[:, self.strand]).sum(dim=1)
        return score[:, None]


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def channel_first(sequence):
    array = as_numpy(sequence)
    if array.ndim == 3:
        array = array[0]
    if array.shape[0] == 4:
        return array
    if array.shape[-1] == 4:
        return array.T
    raise ValueError("Expected a 4-channel one-hot sequence")


def dinucleotide_frequencies(X):
    bases = channel_first(X).argmax(axis=0)
    counts = np.zeros((4, 4), dtype=np.float64)
    for left, right in zip(bases[:-1], bases[1:]):
        counts[left, right] += 1
    total = counts.sum()
    return counts / total if total else counts


def reference_banks(X, seeds, n_references):
    return {
        seed: dinucleotide_shuffle(X.cpu(), n=n_references, random_state=seed)[0].float()
        for seed in seeds
    }


def markov_parameters(X, pseudocount=1e-6):
    frequencies = dinucleotide_frequencies(X)
    transition = frequencies + pseudocount
    transition /= transition.sum(axis=1, keepdims=True)
    initial = channel_first(X).mean(axis=1) + pseudocount
    initial /= initial.sum()
    return initial, transition


def sample_markov_sequences(X, n, random_state=None):
    rng = np.random.default_rng(random_state)
    initial, transition = markov_parameters(X)
    length = channel_first(X).shape[1]
    samples = np.zeros((n, 4, length), dtype=np.float32)
    for i in range(n):
        base = rng.choice(4, p=initial)
        samples[i, base, 0] = 1
        for position in range(1, length):
            base = rng.choice(4, p=transition[base])
            samples[i, base, position] = 1
    return torch.tensor(samples)


def gradient_x_input(model, X, device="cpu"):
    original_device = next(model.parameters()).device
    was_training = model.training
    try:
        model.to(device).eval()
        Xi = X.to(device).float().requires_grad_(True)
        score = model(Xi)
        if isinstance(score, tuple):
            score = score[0]
        if score.ndim > 1:
            score = score.reshape(score.shape[0], -1).sum(dim=1)
        gradient = torch.autograd.grad(score.sum(), Xi)[0]
        return (gradient * Xi).detach().cpu()
    finally:
        if was_training:
            model.train()
        model.to(original_device)


def point_ism(model, X, start, end, batch_size=32, device="cpu"):
    scores = saturation_mutagenesis(
        model,
        X,
        start=start,
        end=end,
        batch_size=batch_size,
        device=device,
    )
    scores = torch.as_tensor(scores).detach().cpu()
    if scores.ndim == 3:
        return scores
    return scores.reshape(X.shape[0], X.shape[1], end - start)


def _model_scores(model, X, batch_size, device):
    values = predict(model=model, X=X, batch_size=batch_size, device=device)
    if isinstance(values, tuple):
        values = values[0]
    values = torch.as_tensor(values)
    return values.reshape(values.shape[0], -1).sum(dim=1).detach().cpu().numpy()


def window_ism(
    model,
    X,
    start,
    end,
    width,
    stride,
    n_replacements,
    random_state,
    batch_size=32,
    device="cpu",
):
    if width < 1 or stride < 1 or n_replacements < 1:
        raise ValueError("width, stride, and n_replacements must be positive")
    positions = np.arange(start, end - width + 1, stride, dtype=int)
    original = _model_scores(model, X, batch_size, device)[0]
    replacements = sample_markov_sequences(
        X, len(positions) * n_replacements, random_state=random_state
    )
    mutants = X.cpu().repeat(len(positions) * n_replacements, 1, 1).float()
    cursor = 0
    for position in positions:
        for _ in range(n_replacements):
            mutants[cursor, :, position : position + width] = replacements[
                cursor, :, position : position + width
            ]
            cursor += 1
    mutant_scores = _model_scores(model, mutants, batch_size, device)
    mutant_scores = mutant_scores.reshape(len(positions), n_replacements).mean(axis=1)
    return positions, original - mutant_scores


def window_scores_to_positions(positions, scores, length, width):
    positions = np.asarray(positions, dtype=int)
    scores = np.asarray(scores, dtype=np.float32)
    output = np.zeros((*scores.shape[:-1], length), dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)
    for index, position in enumerate(positions):
        output[..., position : position + width] += scores[..., index][..., None]
        counts[position : position + width] += 1
    covered = counts > 0
    output[..., covered] /= counts[covered]
    return output


def rolling_profile_maxima(profiles, windows=(1, 5, 20)):
    profiles = np.asarray(profiles, dtype=np.float32)
    flat = profiles.reshape(profiles.shape[0], -1)
    maxima = {}
    positions = {}
    strands = {}
    for width in windows:
        if width == 1:
            rolled = profiles
        else:
            cumsum = np.cumsum(profiles, axis=-1)
            cumsum = np.pad(cumsum, ((0, 0), (0, 0), (1, 0)))
            rolled = cumsum[:, :, width:] - cumsum[:, :, :-width]
        rolled_flat = rolled.reshape(rolled.shape[0], -1)
        argmax = rolled_flat.argmax(axis=1)
        maxima[width] = rolled_flat[np.arange(rolled_flat.shape[0]), argmax]
        positions[width] = argmax % rolled.shape[-1]
        strands[width] = argmax // rolled.shape[-1]
    return maxima, positions, strands


def profile_summaries(probabilities, count_scaled, windows=(1, 5, 20)):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    count_scaled = np.asarray(count_scaled, dtype=np.float32)
    maxima, positions, strands = rolling_profile_maxima(count_scaled, windows)
    flat_prob = probabilities.reshape(probabilities.shape[0], -1)
    entropy = -(flat_prob * np.log2(np.clip(flat_prob, 1e-12, None))).sum(axis=1)
    result = {
        "profile_entropy": entropy,
        "effective_width": np.exp2(entropy),
    }
    for width in windows:
        result[f"max_{width}bp"] = maxima[width]
        result[f"peak_{width}bp_position"] = positions[width]
        result[f"peak_{width}bp_strand"] = strands[width]
    return result


def fold_consensus(positions, strands, activities):
    positions = np.asarray(positions, dtype=np.float32)
    strands = np.asarray(strands, dtype=np.float32)
    activities = np.asarray(activities, dtype=np.float32)
    rounded_strands = np.rint(strands).astype(int)
    agreement = np.maximum((rounded_strands == 0).mean(axis=0), (rounded_strands == 1).mean(axis=0))
    return {
        "peak_position_mean": positions.mean(axis=0),
        "peak_position_sd": positions.std(axis=0),
        "peak_strand_agreement": agreement,
        "peak_activity_mean": activities.mean(axis=0),
        "peak_activity_sd": activities.std(axis=0),
    }


def _profile_outputs(model, X, batch_size, device):
    logits, log_counts = predict(model=model, X=X, batch_size=batch_size, device=device)
    logits = as_numpy(logits).astype(np.float32)
    counts = np.exp(as_numpy(log_counts).reshape(logits.shape[0], -1)).sum(axis=1)
    flat = logits.reshape(logits.shape[0], -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    probabilities = np.exp(centered - centered.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    count_scaled = probabilities * counts[:, None]
    profile_score = (centered * probabilities).sum(axis=1)
    return (
        centered.reshape(logits.shape),
        probabilities.reshape(logits.shape),
        count_scaled.reshape(logits.shape),
        counts.astype(np.float32),
        profile_score.astype(np.float32),
    )


def _jsd(reference_probabilities, genomic_probability):
    p = reference_probabilities.reshape(reference_probabilities.shape[0], -1)
    q = genomic_probability.reshape(1, -1)
    midpoint = 0.5 * (p + q)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    midpoint = np.clip(midpoint, 1e-12, None)
    return 0.5 * (p * np.log2(p / midpoint)).sum(axis=1) + 0.5 * (
        q * np.log2(q / midpoint)
    ).sum(axis=1)


def predicted_activity(model, X, references, batch_size=32, device="cpu"):
    combined = torch.cat([X.cpu().float(), references.cpu().float()])
    centered, probabilities, count_scaled, counts, profile_score = _profile_outputs(
        model, combined, batch_size, device
    )
    genomic_centered = centered[:1]
    genomic_probabilities = probabilities[:1]
    genomic_count_scaled = count_scaled[:1]
    reference_probabilities = probabilities[1:]
    reference_count_scaled = count_scaled[1:]
    return {
        "genomic_centered_logits": genomic_centered,
        "genomic_probabilities": genomic_probabilities,
        "genomic_count_scaled": genomic_count_scaled,
        "genomic_counts": counts[0],
        "genomic_profile_score": profile_score[0],
        "reference_centered_logits": centered[1:],
        "reference_probabilities": reference_probabilities,
        "reference_count_scaled": reference_count_scaled,
        "reference_counts": counts[1:],
        "reference_profile_score": profile_score[1:],
        "reference_profile_jsd": _jsd(reference_probabilities, genomic_probabilities[0]),
        "reference_summaries": profile_summaries(
            reference_probabilities, reference_count_scaled, windows=(1, 5, 20)
        ),
    }


def _robust_high_penalty(values):
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad if mad > 0 else values.std()
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return np.maximum((values - median) / scale, 0)


def reference_weight_schemes(probabilities, counts, temperature=1.0):
    probabilities = np.asarray(probabilities, dtype=np.float32).mean(axis=0)
    counts = np.asarray(counts, dtype=np.float32).mean(axis=0)
    maxima, _, _ = rolling_profile_maxima(probabilities, windows=(20,))
    plus_mass = probabilities[:, 0].sum(axis=1)
    minus_mass = probabilities[:, 1].sum(axis=1)
    metrics = np.vstack([maxima[20], np.abs(plus_mass - minus_mass), np.log1p(counts)])
    contamination = np.vstack(
        [
            np.zeros(probabilities.shape[0], dtype=np.float32),
            _robust_high_penalty(metrics[0]) + _robust_high_penalty(metrics[1]),
            _robust_high_penalty(metrics[0])
            + _robust_high_penalty(metrics[1])
            + _robust_high_penalty(metrics[2]),
        ]
    )
    weights = np.exp(-contamination / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    return {"metrics": metrics, "contamination": contamination, "weights": weights}


def per_reference_attributions(multipliers, X, references):
    multipliers = torch.as_tensor(multipliers).detach().cpu()
    if multipliers.ndim == 4:
        multipliers = multipliers[0]
    references = references.detach().cpu().float()
    expanded = X.detach().cpu().float().expand_as(references)
    projected = hypothetical_attributions((multipliers,), (expanded,), (references,))[0]
    return projected * expanded


def weighted_strand_attributions(per_reference, weights):
    per_reference = np.asarray(per_reference, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if per_reference.shape[0] != weights.shape[0]:
        raise ValueError("weights must match the number of references")
    return np.tensordot(weights, per_reference, axes=(0, 0))


def low_activity_references(metrics, n_references):
    frame = np.asarray(metrics)
    order = np.argsort(frame)
    return order[:n_references]


def reverse_complement_matrix(matrix):
    matrix = np.asarray(matrix)
    return matrix[[3, 2, 1, 0], ::-1]


def reverse_complement_tracks(plus, minus):
    return -np.asarray(minus)[::-1], -np.asarray(plus)[::-1]


def reverse_complement_peak_coordinates(positions, strands, width, window):
    positions = np.asarray(positions)
    strands = np.asarray(strands)
    return width - window - positions, 1 - strands


def genomic_offsets(center, start, end, in_window):
    input_start = center - in_window // 2
    left = start - input_start
    right = end - input_start
    if left < 0 or right > in_window or right <= left:
        raise ValueError(
            f"Requested interval {start + 1}-{end} is outside the {in_window} bp input"
        )
    return int(left), int(right)


def diagnostic_cache_path(cache_dir, experiment, parameters):
    payload = json.dumps(parameters, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return Path(cache_dir) / f"{experiment}.schema{CACHE_SCHEMA_VERSION}.{digest}.npz"
