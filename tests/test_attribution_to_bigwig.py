import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.bpnet.attribute.attribution_to_bigwig import (
    iter_base_intervals,
    iter_averaged_intervals,
    make_windows,
    observed_attribution,
    write_bigwig,
)


def test_observed_attribution_sums_only_ohe_channel():
    attrs = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ]
    )
    ohe = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )

    np.testing.assert_array_equal(observed_attribution(attrs, ohe), [[1.0, 5.0, 12.0]])


def test_iter_averaged_intervals_averages_overlaps():
    scores = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    windows = {
        "chr1": [
            (0, 4, 0, 4, 0),
            (2, 6, 0, 4, 1),
        ]
    }

    intervals = list(iter_averaged_intervals(windows, scores, ["chr1"]))

    assert intervals == [
        ("chr1", 0, 1, 1.0),
        ("chr1", 1, 2, 2.0),
        ("chr1", 2, 3, 6.5),
        ("chr1", 3, 4, 12.0),
        ("chr1", 4, 5, 30.0),
        ("chr1", 5, 6, 40.0),
    ]


def test_iter_base_intervals_keeps_one_base_resolution():
    scores = np.array(
        [
            [1.0, 1.0, 0.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    windows = {
        "chr1": [
            (0, 4, 0, 4, 0),
            (2, 6, 0, 4, 1),
        ]
    }

    intervals = list(iter_base_intervals(windows, scores, ["chr1"]))

    assert intervals == [
        ("chr1", 0, 1, 1.0),
        ("chr1", 1, 2, 1.0),
        ("chr1", 2, 3, 5.0),
        ("chr1", 3, 4, 12.0),
        ("chr1", 4, 5, 30.0),
        ("chr1", 5, 6, 40.0),
    ]


def test_make_windows_clips_chromosome_bounds():
    loci = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [2]})
    scores = np.ones((1, 4))

    windows = make_windows(loci, scores, {"chr1": 10}, in_window=4)

    assert windows == {"chr1": [(0, 3, 1, 4, 0)]}


@pytest.mark.skipif(
    importlib.util.find_spec("pybigtools") is None,
    reason="pybigtools is not installed",
)
def test_write_bigwig_round_trip(tmp_path):
    import pybigtools

    output = tmp_path / "test.bigWig"
    chroms = {"chr1": 8}
    intervals = [
        ("chr1", 0, 2, 1.5),
        ("chr1", 4, 6, -2.0),
    ]

    write_bigwig(output, chroms, iter(intervals))

    with pybigtools.open(str(output)) as bw:
        values = bw.values("chr1", 0, 8, missing=0.0)

    np.testing.assert_array_equal(values, [1.5, 1.5, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0])
