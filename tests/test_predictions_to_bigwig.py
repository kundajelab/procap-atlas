import numpy as np

from src.bpnet.benchmark.predictions_to_bigwig import prediction_to_strand_scores


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
