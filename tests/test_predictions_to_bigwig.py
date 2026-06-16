import numpy as np

from src.bpnet.predict.generate_predicted_tracks import (
    prediction_to_strand_scores,
    scale_profile_logits,
)


def test_scale_profile_logits_rescales_softmax_by_exp_counts():
    logits = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    log_counts = np.array([[np.log(8.0)]])

    scaled = scale_profile_logits(logits, log_counts)

    np.testing.assert_allclose(scaled, [[[2.0, 2.0], [2.0, 2.0]]])


def test_prediction_to_strand_scores_channel_first_negates_minus():
    pred = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ]
    )

    plus, minus = prediction_to_strand_scores(pred)

    np.testing.assert_array_equal(plus, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(minus, [[-4.0, -5.0, -6.0]])


def test_prediction_to_strand_scores_channel_last():
    pred = np.array(
        [
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ]
        ]
    )

    plus, minus = prediction_to_strand_scores(pred)

    np.testing.assert_array_equal(plus, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(minus, [[-4.0, -5.0, -6.0]])
