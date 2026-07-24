import numpy as np

from src.bpnet.predict.generate_predicted_tracks import (
    ensemble_fold_predictions,
    prediction_to_strand_scores,
    scale_profile_logits,
)


def test_scale_profile_logits_rescales_softmax_by_exp_counts():
    logits = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    log_counts = np.array([[np.log(8.0)]])

    scaled = scale_profile_logits(logits, log_counts)

    np.testing.assert_allclose(scaled, [[[2.0, 2.0], [2.0, 2.0]]])


def test_ensemble_fold_predictions_averages_logits_and_log_counts_before_scaling():
    fold_logits = [
        np.array([[[0.0, 0.0], [0.0, 0.0]]]),
        np.array([[[2.0, 0.0], [0.0, 0.0]]]),
    ]
    fold_log_counts = [
        np.array([[np.log(4.0)]]),
        np.array([[np.log(16.0)]]),
    ]

    scaled = ensemble_fold_predictions(fold_logits, fold_log_counts)

    expected_logits = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    expected_log_counts = np.array([[np.log(8.0)]])
    expected = scale_profile_logits(expected_logits, expected_log_counts)
    old_scaled_average = np.mean(
        [
            scale_profile_logits(logits, log_counts)
            for logits, log_counts in zip(fold_logits, fold_log_counts)
        ],
        axis=0,
    )

    np.testing.assert_allclose(scaled, expected)
    assert not np.allclose(scaled, old_scaled_average)


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
