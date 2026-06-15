import numpy as np
import torch
from bpnetlite.attribute import deep_lift_shap

from src.bpnet.attribute.locus_diagnostics import (
    CACHE_SCHEMA_VERSION,
    diagnostic_cache_path,
    dinucleotide_frequencies,
    fold_consensus,
    genomic_offsets,
    gradient_x_input,
    low_activity_references,
    markov_parameters,
    per_reference_attributions,
    point_ism,
    predicted_activity,
    profile_summaries,
    reference_banks,
    reference_weight_schemes,
    rolling_profile_maxima,
    reverse_complement_matrix,
    reverse_complement_peak_coordinates,
    reverse_complement_tracks,
    sample_markov_sequences,
    StrandProfileWrapper,
    weighted_strand_attributions,
    window_ism,
    window_scores_to_positions,
)


class ToyModel(torch.nn.Module):
    def forward(self, X):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device=X.device)
        return (X * weights[None, :, None]).sum(dim=(1, 2), keepdim=True)


class ToyBPNet(torch.nn.Module):
    def forward(self, X):
        logits = torch.stack((X[:, 0] + 2 * X[:, 1], X[:, 2] + 2 * X[:, 3]), dim=1)
        log_counts = X[:, 3].sum(dim=1, keepdim=True).log1p()
        return logits, log_counts


class PositionBPNet(torch.nn.Module):
    def forward(self, X):
        position_weights = torch.linspace(-1, 1, X.shape[-1], device=X.device)
        plus = 3 * X[:, 0] + X[:, 1] * position_weights
        minus = 3 * X[:, 3] - X[:, 2] * position_weights
        log_counts = (
            2 * X[:, 0, : X.shape[-1] // 2].sum(dim=1, keepdim=True)
            / X.shape[-1]
        )
        return torch.stack((plus, minus), dim=1), log_counts


def one_hot(sequence):
    alphabet = {base: i for i, base in enumerate("ACGT")}
    X = torch.zeros((1, 4, len(sequence)), dtype=torch.float32)
    for position, base in enumerate(sequence):
        X[0, alphabet[base], position] = 1
    return X


def test_dinucleotide_frequencies_and_markov_parameters():
    X = one_hot("AACGTT")
    frequencies = dinucleotide_frequencies(X)
    assert np.isclose(frequencies.sum(), 1)
    assert frequencies[0, 0] == 0.2
    assert frequencies[0, 1] == 0.2
    initial, transition = markov_parameters(X)
    assert np.isclose(initial.sum(), 1)
    assert np.allclose(transition.sum(axis=1), 1)


def test_reference_banks_are_seeded_and_preserve_dinucleotides():
    X = one_hot("ACGTTGCAACGT")
    first = reference_banks(X, [7], 4)[7]
    second = reference_banks(X, [7], 4)[7]
    assert torch.equal(first, second)
    expected = dinucleotide_frequencies(X)
    for reference in first:
        assert np.allclose(dinucleotide_frequencies(reference), expected)


def test_low_activity_references_are_deterministic_and_low_ranked():
    X = one_hot("ACGTTGCAACGTACGTTGCAACGT")
    first = low_activity_references(
        X,
        [PositionBPNet(), PositionBPNet()],
        n_candidates=40,
        n_references=5,
        random_state=17,
        batch_size=16,
        device="cpu",
        min_hamming_fraction=0.2,
    )
    second = low_activity_references(
        X,
        [PositionBPNet(), PositionBPNet()],
        n_candidates=40,
        n_references=5,
        random_state=17,
        batch_size=16,
        device="cpu",
        min_hamming_fraction=0.2,
    )

    assert torch.equal(first["references"], second["references"])
    assert np.array_equal(first["selected_indices"], second["selected_indices"])
    assert np.all(
        first["selection_score"][first["selected_indices"]]
        <= np.quantile(first["selection_score"], 0.5)
    )
    expected = dinucleotide_frequencies(X)
    for reference in first["references"]:
        assert np.allclose(dinucleotide_frequencies(reference), expected)
    bases = first["references"].argmax(dim=1).numpy()
    for i in range(len(bases)):
        for j in range(i):
            assert np.mean(bases[i] != bases[j]) >= 0.2


def test_markov_sampling_is_deterministic():
    X = one_hot("AACCGGTTAACCGGTT")
    first = sample_markov_sequences(X, length=8, n=5, random_state=3)
    second = sample_markov_sequences(X, length=8, n=5, random_state=3)
    assert torch.equal(first, second)
    assert first.shape == (5, 4, 8)
    assert torch.all(first.sum(dim=1) == 1)


def test_attribution_methods_with_toy_model():
    X = one_hot("ACGTAC")
    model = ToyModel()
    gradient = gradient_x_input(model, X, device="cpu")
    expected = X * torch.tensor([1.0, 2.0, 3.0, 4.0])[None, :, None]
    assert torch.allclose(gradient, expected)

    ism = point_ism(model, X, 1, 5, batch_size=8, device="cpu")
    assert ism.shape == (1, 4, 4)
    positions, scores = window_ism(
        model,
        X,
        start=0,
        end=6,
        window=2,
        stride=2,
        n_replacements=3,
        random_state=1,
        batch_size=8,
        device="cpu",
    )
    assert positions.tolist() == [0, 2, 4]
    assert scores.shape == (3,)


def test_window_scores_are_averaged_across_overlaps():
    positions = np.array([1, 3])
    scores = np.array([[2.0, 6.0], [-2.0, 2.0]])
    projected = window_scores_to_positions(
        positions, scores, length=7, window=3
    )

    assert np.allclose(projected[0], [0, 2, 2, 4, 6, 6, 0])
    assert np.allclose(projected[1], [0, -2, -2, 0, 2, 2, 0])


def test_reverse_complement_and_offsets():
    matrix = np.arange(24).reshape(4, 6)
    assert np.array_equal(
        reverse_complement_matrix(matrix), matrix[[3, 2, 1, 0], ::-1]
    )
    plus = np.array([1, 2, 3])
    minus = np.array([-4, -5, -6])
    rc_plus, rc_minus = reverse_complement_tracks(plus, minus)
    assert np.array_equal(rc_plus, [6, 5, 4])
    assert np.array_equal(rc_minus, [-3, -2, -1])
    rc_positions, rc_strands = reverse_complement_peak_coordinates(
        np.array([0, 7]), np.array([0, 1]), length=10, window=3
    )
    assert np.array_equal(rc_positions, [7, 0])
    assert np.array_equal(rc_strands, [1, 0])
    assert genomic_offsets(100, 95, 105, 20) == (5, 15)


def test_profile_outputs_are_centered_and_count_scaled():
    genomic = one_hot("ACGTAC" * 4)
    references = torch.cat([one_hot("TGCATG" * 4), one_hot("AAAACC" * 4)])
    result = predicted_activity(
        ToyBPNet(), genomic, references, batch_size=3, device="cpu"
    )
    centered = result["reference_centered_logits"]
    probabilities = result["reference_probabilities"]
    scaled = result["reference_count_scaled"]
    assert np.allclose(centered.reshape(2, -1).mean(axis=1), 0, atol=1e-7)
    assert np.allclose(probabilities.reshape(2, -1).sum(axis=1), 1)
    assert np.allclose(scaled.reshape(2, -1).sum(axis=1), result["reference_counts"])


def test_strand_profile_wrapper_uses_joint_softmax_weights():
    X = one_hot("ACGTAC")
    model = ToyBPNet()
    logits = model(X)[0]
    centered = logits.flatten(start_dim=1)
    centered = centered - centered.mean(dim=-1, keepdim=True)
    weighted = (centered * torch.softmax(centered, dim=-1)).reshape_as(logits)

    plus = StrandProfileWrapper(model, 0)(X)
    minus = StrandProfileWrapper(model, 1)(X)

    assert torch.allclose(plus[:, 0], weighted[:, 0].sum(dim=-1))
    assert torch.allclose(minus[:, 0], weighted[:, 1].sum(dim=-1))
    assert torch.allclose(
        plus + minus,
        (centered * torch.softmax(centered, dim=-1)).sum(dim=-1, keepdim=True),
    )


def test_reference_weights_are_uniform_for_equal_activity():
    probabilities = np.full((2, 4, 2, 20), 1 / 40)
    counts = np.full((2, 4), 5.0)
    result = reference_weight_schemes(probabilities, counts)

    assert np.allclose(result["weights"], 0.25)
    assert np.all(result["weights"] >= 0)
    assert np.allclose(result["weights"].sum(axis=1), 1)


def test_reference_weights_penalize_profile_spikes_and_counts():
    probabilities = np.full((2, 5, 2, 20), 1 / 40)
    probabilities[:, 4] = 0
    probabilities[:, 4, 0, :20] = 1 / 20
    counts = np.ones((2, 5))
    counts[:, 3] = 50

    result = reference_weight_schemes(probabilities, counts)
    uniform, profile_only, profile_counts = result["weights"]

    assert np.allclose(uniform, 0.2)
    assert profile_only[4] < profile_only[0]
    assert np.isclose(profile_only[3], profile_only[0])
    assert profile_counts[3] < profile_only[3]


def test_weighted_strand_attributions_preserve_weighted_delta():
    X = one_hot("ACGTAC")
    references = torch.cat(
        [one_hot("TGCATG"), one_hot("AAAACC"), one_hot("CCGGTT")]
    )
    weights = np.array([0.2, 0.3, 0.5])
    model = ToyBPNet()
    strand_attributions = []
    strand_deltas = []

    for strand in range(2):
        wrapper = StrandProfileWrapper(model, strand)
        multipliers = deep_lift_shap(
            wrapper,
            X,
            references=references[None],
            raw_outputs=True,
            hypothetical=True,
            device="cpu",
        )
        per_reference = per_reference_attributions(
            multipliers, X, references
        ).numpy()
        strand_attributions.append(per_reference)
        with torch.no_grad():
            genomic_score = wrapper(X)[0, 0].item()
            reference_scores = wrapper(references)[:, 0].numpy()
        strand_deltas.append(genomic_score - reference_scores)

    strand_attributions = np.asarray(strand_attributions)
    weighted = np.asarray(
        [
            weighted_strand_attributions(values, weights)
            for values in strand_attributions
        ]
    )
    weighted_delta = np.asarray(strand_deltas) @ weights

    assert np.allclose(weighted.sum(axis=(1, 2)), weighted_delta, atol=1e-5)
    assert np.isclose(weighted.sum(), weighted_delta.sum(), atol=1e-5)


def test_profile_summaries_and_fold_consensus():
    profiles = np.zeros((2, 2, 8), dtype=float)
    profiles[0, 1, 3:6] = [1, 2, 1]
    profiles[1, 0, 1:4] = [2, 2, 2]
    maxima, positions, strands = rolling_profile_maxima(profiles, windows=(1, 3))
    assert maxima[3].tolist() == [4, 6]
    assert positions[3].tolist() == [3, 1]
    assert strands[3].tolist() == [1, 0]
    probabilities = profiles / profiles.reshape(2, -1).sum(axis=1)[:, None, None]
    summaries = profile_summaries(probabilities, profiles, windows=(1, 3))
    assert summaries["peak_3bp_position"].tolist() == [3, 1]
    assert np.all(summaries["effective_width"] > 1)

    consensus = fold_consensus(
        np.array([[3, 1], [4, 1], [3, 2]]),
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[4, 6], [5, 6], [3, 3]]),
    )
    assert np.allclose(consensus["peak_strand_agreement"], [1, 2 / 3])
    assert consensus["peak_position_sd"][0] > 0


def test_cache_schema_changes_cache_key(tmp_path):
    first = diagnostic_cache_path(tmp_path, "test", {"schema": 1})
    second = diagnostic_cache_path(
        tmp_path, "test", {"schema": CACHE_SCHEMA_VERSION}
    )
    assert first != second


def test_float16_profile_cache_round_trip(tmp_path):
    path = tmp_path / "profiles.npz"
    profiles = np.linspace(0, 1, 40, dtype=np.float32).reshape(2, 2, 10)
    np.savez_compressed(path, profiles=profiles.astype(np.float16))
    loaded = np.load(path)["profiles"]
    assert loaded.dtype == np.float16
    assert np.allclose(loaded, profiles, atol=5e-4)
