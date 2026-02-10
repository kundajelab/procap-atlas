"""
Adapts functions from tangermeme/match.py to extract GC-matched negatives to
allow for thresholding on multiple bigwig files. Can be run as a script, but
main use in this project is to be called from `gc_match_run.py`, which calls
it for each experiment.
"""

import argparse

import numpy
import pandas
import pybigtools
import pyfaidx
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from tangermeme.match import (
    _calculate_char_perc,
    _char_perc_from_coords,
    _counts_from_coords,
    _loci_coords_generator,
    _resize_coords_generator,
    _valid_generator,
)
from tqdm import tqdm


def _extract_and_filter_chrom(
    fasta,
    chrom,
    in_window,
    out_window,
    max_n_perc=0.1,
    gc_bin_width=0.02,
    bigwig=None,
    signal_threshold=None,
):
    """Calculate GC content for, and filter, one chromosome.

    This function will take in the name of a FASTA file, a chromosome, and a
    percentage of Ns that cannot be exceeded, and return a set of passing loci
    and their exact GC percentage. Optionally, it will also take in a bigwig
    and filter the loci based on having a signal threshold smaller than some
    value, where the signal is summed across the width.


    Parameters
    ----------
    fasta: str
            The filepath to the FASTA file to extract sequences from.

    chrom: str
            The chromosome to extract from the FASTA file. Must be in the file.

    in_window: int
            The window to calculate the GC content over, corresponding to the input
            window of the downstream model that will be trained.

    out_window: int
            The window to calculate signal for and apply the signal threshold to,
            corresponding to the output window of the downstream model that will
            be trained.

    max_n_perc: float, range=(0, 1.0), optional
            The maximum percentage of N characters in each window to be considered.
            All windows with a higher percentage are discarded. Default is 0.1.

    gc_bin_width: float, range=(0, 1.0), optional
            The bin size to discretize GC content. Default is 0.02.

    bigwig: str or None, optional
            If filtering regions based on signal strength, calculate the signal
            from the sum of provided bigwig(s). If None, do not filter based on signal
            strength. Default is None.

    signal_threshold: float or None, optional
            The maximum possible signal, summed across the entire window, that
            each window can have without being filtered. If the window has a summed
            signal higher than this value, the window is discarded. Default is None.

    Returns
    -------
    gc_percs: dict
            A dictionary where the keys are observed GC bins and the values are
            lists of loci that are in that GC bin. Each returned locus is the index
            not the true value and so corresponds to the real position integer
            divided by in_window.
    """

    with pyfaidx.Fasta(fasta) as f:
        sequence = f[chrom][:].seq.upper()

    gc_perc = _calculate_char_perc(sequence, in_window, "GC")
    n_perc = _calculate_char_perc(sequence, in_window, "N")
    del sequence

    idxs = n_perc <= max_n_perc

    if bigwig is not None:
        assert in_window >= out_window, "out_window cannot be larger than in_window."
        left_flank = (in_window - out_window) // 2
        right_flank = (in_window - out_window + 1) // 2

        if isinstance(bigwig, str):
            bigwig = [bigwig]
        values = []
        for b in bigwig:
            with pybigtools.open(b, "r") as bw:
                try:
                    values.append(
                        numpy.abs(numpy.nan_to_num(bw.values(chrom, 0, None)))
                    )
                except RuntimeError:
                    return {}

        values = [v[: v.shape[0] // in_window * in_window] for v in values]
        values = [v.reshape(-1, in_window) for v in values]
        values = [v[:, left_flank:-right_flank] for v in values]
        values = sum([numpy.nansum(v, axis=-1) for v in values])

        idxs = idxs & (values <= signal_threshold)

    gc_perc = ((gc_perc + gc_bin_width / 2.0) // gc_bin_width).astype(int)
    unique_gc = numpy.unique(gc_perc[idxs]).tolist()
    gc_perc = {
        gc: numpy.nonzero(idxs & (gc_perc == gc))[0].tolist() for gc in unique_gc
    }

    return gc_perc


def extract_matching_loci(
    loci,
    fasta,
    in_window=2114,
    out_window=1000,
    max_n_perc=0.1,
    gc_bin_width=0.02,
    bigwig=None,
    signal_beta=0.5,
    chroms=None,
    random_state=None,
    n_jobs=1,
    verbose=False,
):
    """Extract matching loci given a fasta file.

    This function takes in a set of loci (a bed file or a pandas dataframe in
    bed format) and returns a GC-matched set of negatives. This will also
    perform basic filtering to ignore regions of the genome that are too high
    in Ns. Optionally, it can take in a list of bigwigs and a signal threshold
    and only select regions that have fewer than a threshold of counts in each
    region.

    Importantly, it will apply `max_n_perc` to both the loci that are passed in
    and also potential regions that can be selected. This means that if a locus
    passed in has higher than `max_n_perc` number of unspecified positions,
    it will be filtered out, and a smaller number of positions will be selected.
    This is done because the GC content of a region with many Ns in it is not
    trustworthy.


    Parameters
    ----------
    loci: str or pandas dataframe
            A filepath to a bed file, or a pandas dataframe in bed format.

    fasta: str
            The filepath to the FASTA file to extract sequences from.

    in_window: int
            The window to calculate the GC content over, corresponding to the input
            window of the downstream model that will be trained.

    out_window: int
            The window to calculate signal for and apply the signal threshold to,
            corresponding to the output window of the downstream model that will
            be trained.

    max_n_perc: float, range=(0, 1.0), optional
            The maximum percentage of N characters in each window to be considered.
            All windows with a higher percentage are discarded. Default is 0.1.

    gc_bin_width: float, range=(0, 1.0), optional
            The bin size to discretize GC content. Default is 0.02.

    bigwig: str, list of str or None, optional
            If filtering regions based on signal strength, calculate the signal
            from this bigwig(s). If None, do not filter based on signal strength.
            Default is None.

    signal_beta: float or None, optional
            A multiplier of the robust minimum signal calculated from `loci` that
            each background region must have fewer reads then. Only relevant if a
            bigwig is passed in. Default is 0.5.

    chroms: list, tuple, or None, optional
            A set of chromosomes to use when choosing matching loci. If None, only
            use chromosomes that the loci themselves are drawn from. Default is
            None.

    random_state: numpy.random.RandomState, int or None, optional
            A random state to use for sampling loci. If a RandomState object or
            an integer, this will produce deterministic sampling. If None, sampling
            will be different each time. Default is None.

    n_jobs: integer, optional
            Number of parallel processes to use for extracting background gc content.
            -1 means use all available CPUs. Default is 1.

    verbose: bool, optional
            Whether to print display bars and diagnostics to ensure that the
            sampling is reasonable. When set to True, there may be a large amount
            of output. Default is False.


    Returns
    -------
    matched_loci: pandas.DataFrame
            A bed-formatted set of matched loci sorted first by chromosome and
            then by position on the chromosome. Note that these are not sorted
            such that the i-th position in this file is a GC match for the i-th
            position in the original locus file.
    """

    if not isinstance(random_state, numpy.random.RandomState):
        random_state = numpy.random.RandomState(random_state)

    if isinstance(loci, str):
        loci = pandas.read_csv(
            loci,
            sep="\t",
            usecols=[0, 1, 2],
            header=None,
            index_col=False,
            names=["chrom", "start", "end"],
        )

    if chroms is not None:
        loci = loci[numpy.isin(loci["chrom"], chroms)]
    else:
        chroms = numpy.unique(loci["chrom"])

    fa = pyfaidx.Fasta(fasta)
    chrom_sizes = {key: len(fa[key]) for key in chroms}

    if verbose:
        print("Processing given loci.")

    coords = _loci_coords_generator(loci, max(in_window, out_window))
    coords = list(_valid_generator(coords, chrom_sizes))
    num_regions = len(coords)

    threshold = None
    if bigwig is not None:
        coords = list(_resize_coords_generator(coords, out_window))
        if isinstance(bigwig, str):
            bigwig = [bigwig]
        loci_count = sum(
            [
                numpy.abs(
                    numpy.nan_to_num(
                        _counts_from_coords(
                            b, coords, num_regions, buffer=False, verbose=verbose
                        )
                    )
                )
                for b in bigwig
            ]
        )
        robust_min = numpy.nanquantile(loci_count, 0.01).item()
        threshold = robust_min * signal_beta

    coords = list(_resize_coords_generator(coords, in_window))
    loci_n = _char_perc_from_coords(
        fasta, coords, "N", num_regions, buffer=False, verbose=verbose
    )
    loci_gc = _char_perc_from_coords(
        fasta, coords, "GC", num_regions, buffer=False, verbose=verbose
    )
    loci_gc = loci_gc[loci_n < max_n_perc]

    loci_gc = ((loci_gc + gc_bin_width / 2.0) // gc_bin_width).astype(int)
    loci_bin_count = numpy.zeros(int(1.0 / gc_bin_width) + 1, dtype=int)
    for gc_bin in loci_gc:
        loci_bin_count[gc_bin] += 1

    # Extract mask of already-selected loci
    mask = {chrom: [] for chrom in chroms}
    for locus in loci.itertuples(index=False):
        if locus.chrom not in mask:
            continue

        start = locus.start // in_window
        end = locus.end // in_window + 1

        mask[locus.chrom].extend(range(start, end))

    for chrom, values in mask.items():
        mask[chrom] = set(values)

    # Get GC content of background regions
    desc = "Getting background GC"
    f = delayed(_extract_and_filter_chrom)
    chrom_percs = Parallel(n_jobs=n_jobs)(
        f(
            fasta=fasta,
            chrom=chrom,
            in_window=in_window,
            out_window=out_window,
            max_n_perc=max_n_perc,
            gc_bin_width=gc_bin_width,
            bigwig=bigwig,
            signal_threshold=threshold,
        )
        for chrom in tqdm(chroms, disable=not verbose, desc=desc)
    )

    # Merge them into a single dictionary, keeping track of chroms
    bg_bin_count = numpy.zeros(int(1.0 / gc_bin_width) + 1, dtype=int)
    gc_percs = {perc: [] for perc in range(len(bg_bin_count))}

    for chrom, percs in zip(chroms, chrom_percs):
        for key, values in percs.items():
            for value in values:
                if value not in mask[chrom]:
                    gc_percs[key].append((chrom, value))
                    bg_bin_count[key] += 1

    for key, value in gc_percs.items():
        random_state.shuffle(value)

    orig_bg_bin_count = bg_bin_count.copy()
    orig_loci_bin_count = loci_bin_count.copy()

    # Match the sizes
    matched_loci_bin_count = numpy.minimum(bg_bin_count, loci_bin_count)
    bg_bin_count -= matched_loci_bin_count
    loci_bin_count -= matched_loci_bin_count

    n = len(loci_bin_count)
    for i in range(n - 1, -1, -1):
        if loci_bin_count[i] == 0:
            continue

        for offset in range(n):
            idx = i + offset
            if idx < n:
                count = min(bg_bin_count[idx], loci_bin_count[i])
                bg_bin_count[idx] -= count
                loci_bin_count[i] -= count
                matched_loci_bin_count[idx] += count

            if loci_bin_count[i] == 0:
                break

            idx = i - offset
            if idx > 0:
                count = min(bg_bin_count[idx], loci_bin_count[i])
                bg_bin_count[idx] -= count
                loci_bin_count[i] -= count
                matched_loci_bin_count[idx] += count

            if loci_bin_count[i] == 0:
                break

    if verbose:
        numpy.set_printoptions(suppress=True)
        print("GC Bin\tBackground Count\tPeak Count\tChosen Count")
        for i in range(n):
            print(
                "{:2.2f}: {:8d}\t{:8d}\t{:8d}".format(
                    numpy.arange(0, 1.01, gc_bin_width)[i],
                    orig_bg_bin_count[i],
                    orig_loci_bin_count[i],
                    matched_loci_bin_count[i],
                )
            )

    # Extract the loci
    matched_loci = {"chrom": [], "start": [], "end": []}
    for i in range(n):
        for j in range(matched_loci_bin_count[i]):
            chrom, start = gc_percs[i][j]

            matched_loci["chrom"].append(chrom)
            matched_loci["start"].append(start * in_window)
            matched_loci["end"].append((start + 1) * in_window)

    matched_loci = pandas.DataFrame(matched_loci)

    if verbose:
        matched_gc = []
        for i, j in enumerate(matched_loci_bin_count):
            matched_gc.extend([i] * j)

        stats = ks_2samp(loci_gc, matched_gc)
        print(
            "GC-bin KS test stat:{:3.3}, p-value {:3.3}".format(
                stats.statistic, stats.pvalue
            )
        )

        if bigwig is not None:
            print("Processing matched loci.")
            coords = _loci_coords_generator(matched_loci, out_window)
            num_regions = len(matched_loci)
            matched_count_max = sum(
                numpy.abs(
                    numpy.nan_to_num(
                        _counts_from_coords(
                            b, coords, num_regions, buffer=False, verbose=verbose
                        )
                    )
                )
                for b in bigwig
            ).max()
            print("Peak Robust Signal Minimum: {}".format(robust_min))
            print("Matched Signal Maximum: {}".format(matched_count_max))

    matched_loci = matched_loci.sort_values(["chrom", "start"])
    return matched_loci


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("-i", "--peaks", required=True, help="Peak bed file.")
    parser.add_argument("-f", "--fasta", required=True, help="Genome FASTA file.")
    parser.add_argument("-o", "--output", required=True, help="Output bed file.")
    parser.add_argument("-b", "--bigwigs", nargs="+", help="Optional signal bigwigs.")
    parser.add_argument(
        "-l", "--bin_width", type=float, default=0.02, help="GC bin width to match."
    )
    parser.add_argument(
        "-n",
        "--max_n_perc",
        type=float,
        default=0.1,
        help="Maximum percentage of Ns allowed in each locus.",
    )
    parser.add_argument(
        "-a",
        "--beta",
        type=float,
        default=0.5,
        help="Multiplier on the minimum counts in peaks.",
    )
    parser.add_argument(
        "-w",
        "--in_window",
        type=int,
        default=2114,
        help="Width for calculating GC content.",
    )
    parser.add_argument(
        "-x",
        "--out_window",
        type=int,
        default=1000,
        help="Non-overlapping stride to use for loci.",
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    args = parser.parse_args()

    chroms = list(pyfaidx.Fasta(args.fasta).keys())

    # Extract regions that match the GC content of the peaks
    matched_loci = extract_matching_loci(
        loci=args.peaks,
        fasta=args.fasta,
        gc_bin_width=args.bin_width,
        max_n_perc=args.max_n_perc,
        bigwig=args.bigwigs,
        signal_beta=args.beta,
        in_window=args.in_window,
        out_window=args.out_window,
        chroms=chroms,
        verbose=args.verbose,
        n_jobs=1,
    )

    matched_loci.to_csv(args.output, header=False, sep="\t", index=False)


if __name__ == "__main__":
    main()
