"""Small shared helpers for working with regions.npz's raw arrays, used by
several hit-scoring/diagnostic scripts.
"""


def build_peak_row_index(peaks_df):
    """Map peak_id -> row index into the regions.npz contribution array."""
    peak_ids = peaks_df["peak_id"].to_numpy()
    return {int(pid): i for i, pid in enumerate(peak_ids)}


def project_contribs(contribs, sequences):
    """Collapse hypothetical (N, 4, L) contribs to projected (N, L) by
    masking with the one-hot sequence, matching finemo.main.report()'s own
    reshape. No-op if contribs is already (N, L).
    """
    if contribs.ndim == 2:
        return contribs
    return (contribs * sequences).sum(axis=1)
