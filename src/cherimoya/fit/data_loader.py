# data_loader.py
#
# Thin wrapper around cherimoya's own PeakGenerator/PeakNegativeSampler so
# this repo stays in sync with upstream (multi-group signal support, RC
# channel permutation, per-group outlier filtering) instead of maintaining a
# frozen fork of it. The one thing layered on top: this repo's PRO-cap
# bigWigs store minus-strand signal as negative values (confirmed), and
# upstream cherimoya's PeakNegativeSampler no longer un-negates signal
# in __getitem__ (unlike the pre-refactor version this file used to mirror),
# so it's done here instead -- matching the explicit torch.abs(y_valid)
# already applied to validation data in fit_cherimoya.py.

import torch
from cherimoya.io import PeakGenerator as _PeakGenerator
from cherimoya.io import PeakNegativeSampler as _PeakNegativeSampler


class _AbsPeakNegativeSampler(_PeakNegativeSampler):
    """PeakNegativeSampler that un-negates minus-strand signal on return."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if len(item) == 4:
            Xi, Xi_ctl, yi, is_peak = item
            return Xi, torch.abs(Xi_ctl), torch.abs(yi), is_peak
        Xi, yi, is_peak = item
        return Xi, torch.abs(yi), is_peak


def PeakGenerator(*args, **kwargs):
    """cherimoya.io.PeakGenerator, with minus-strand signal un-negated.

    Identical to cherimoya's own PeakGenerator in every other respect --
    see its docstring (`cherimoya.io.PeakGenerator`) for the full parameter
    list, including the ``signals``/``controls`` grouping convention.
    """
    loader = _PeakGenerator(*args, **kwargs)
    # `dataset[idx]` resolves __getitem__ on the type, not the instance, so
    # assigning over `loader.dataset.__getitem__` directly would silently
    # have no effect. Swapping __class__ works because the subclass adds no
    # new __init__-set state, only a method override.
    loader.dataset.__class__ = _AbsPeakNegativeSampler
    return loader
