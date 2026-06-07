import numpy as np
import torch

from src.bpnet.attribute.locus_diagnostics import (
    CACHE_SCHEMA_VERSION,
    diagnostic_cache_path,
    dinucleotide_frequencies,
    fold_consensus,
    genomic_offsets,
    gradient_x_input,
    markov_parameters,
    point_ism,
    predicted_activity,
    profile_summaries,
    reference_banks,
    rolling_profile_maxima,
    reverse_complement_matrix,
    reverse_complement_peak_coordinates,
    reverse_complement_tracks,
    sample_markov_sequences,
    window_ism,
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
