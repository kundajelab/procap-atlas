import numpy as np
import torch
from tangermeme.deep_lift_shap import deep_lift_shap

from src.bpnet.attribute.attribute_bpnet import nucleotide_frequency_references
from src.modeling.profile import (
    count_scaled_profile,
    orientation_index,
    strand_masses,
)
from src.modeling.wrappers import OrientationIndexWrapper


class ToyBPNet(torch.nn.Module):
    def forward(self, X):
        logits = torch.stack((2 * X[:, 0] + X[:, 1], X[:, 2] + 2 * X[:, 3]), dim=1)
        log_counts = X[:, 0].sum(dim=-1, keepdim=True).log1p()
        return logits, log_counts


def one_hot(sequence):
    alphabet = {base: i for i, base in enumerate("ACGT")}
    X = torch.zeros((1, 4, len(sequence)), dtype=torch.float32)
    for position, base in enumerate(sequence):
        X[0, alphabet[base], position] = 1
    return X


def test_uniform_logits_orientation_index_is_half():
    logits = torch.zeros((3, 2, 5))

    orientation = orientation_index(logits, is_logit=True)

    assert orientation.shape == (3, 1)
    assert torch.allclose(orientation, torch.full((3, 1), 0.5))


def test_orientation_index_matches_direct_max_for_each_dominant_strand():
    logits = torch.tensor(
        [
            [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
        ]
    )
    masses = strand_masses(logits)
    expected = masses.max(dim=1, keepdim=True).values

    orientation = orientation_index(logits, is_logit=True)

    assert torch.allclose(orientation, expected)
    assert masses[0, 0] > masses[0, 1]
    assert masses[1, 1] > masses[1, 0]


def test_count_scaled_orientation_equals_probability_orientation():
    logits = np.array(
        [
            [[2.0, 1.0, 0.0], [0.0, 1.0, 1.5]],
            [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        ]
    )
    log_counts = np.log(np.array([[10.0], [3.0]]))
    scaled = count_scaled_profile(logits, log_counts)
    scaled_mass = scaled.sum(axis=-1)
    scaled_orientation = scaled_mass.max(axis=1, keepdims=True) / scaled_mass.sum(
        axis=1, keepdims=True
    )

    np.testing.assert_allclose(
        scaled_orientation,
        orientation_index(logits, is_logit=True),
    )


def test_orientation_index_from_observed_profile_values():
    profile = torch.tensor(
        [
            [[3.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [4.0, 2.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    orientation = orientation_index(profile)

    expected = torch.tensor([[4.0 / 6.0], [7.0 / 8.0], [0.0]])
    assert torch.allclose(orientation, expected)


def test_count_scaled_profile_matches_manual_softmax_scaling():
    logits = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    log_counts = np.array([[np.log(8.0)]])

    scaled = count_scaled_profile(logits, log_counts)

    np.testing.assert_allclose(scaled, [[[2.0, 2.0], [2.0, 2.0]]])


def test_orientation_index_wrapper_matches_shared_helper():
    X = one_hot("ACGTAC")
    model = ToyBPNet()
    logits = model(X)[0]
    wrapper = OrientationIndexWrapper(model)

    wrapped = wrapper(X)

    assert wrapped.shape == (1, 1)
    assert torch.allclose(wrapped, orientation_index(logits, is_logit=True))


def test_orientation_index_wrapper_runs_deeplift_shap_on_cpu():
    X = one_hot("ACGTAC")
    references = torch.cat([one_hot("TGCATG"), one_hot("AAAACC")])
    wrapper = OrientationIndexWrapper(ToyBPNet())

    attributions = deep_lift_shap(
        wrapper,
        X,
        references=references[None],
        hypothetical=True,
        device="cpu",
    )

    assert attributions.shape == X.shape
    assert torch.isfinite(attributions).all()


def test_nucleotide_frequency_references_are_soft_pfms():
    X = torch.cat([one_hot("AAAACC"), one_hot("GGTTTT")])

    first = nucleotide_frequency_references(X, n=3, random_state=7)
    second = nucleotide_frequency_references(X, n=3, random_state=7)

    assert first.shape == (2, 3, 4, 6)
    assert torch.equal(first, second)
    assert torch.all(first.sum(dim=2) == 1)
    np.testing.assert_allclose(first[0, 0, :, 0].numpy(), [4 / 6, 2 / 6, 0, 0])
    np.testing.assert_allclose(first[1, 0, :, 0].numpy(), [0, 0, 2 / 6, 4 / 6])
