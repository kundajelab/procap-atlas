import torch
from bpnetlite.io import PeakNegativeSampler
from tangermeme.io import extract_loci


def PeakGenerator(
    peaks,
    negatives,
    sequences,
    signals,
    controls=None,
    chroms=None,
    in_window=2114,
    out_window=1000,
    max_jitter=128,
    negative_ratio=0.1,
    reverse_complement=True,
    shuffle=True,
    min_counts=None,
    max_counts=None,
    summits=False,
    exclusion_lists=None,
    random_state=None,
    pin_memory=True,
    num_workers=0,
    batch_size=32,
    verbose=False,
):
    """This is a constructor function that handles all IO.

    This function will extract signal from all signal and control files,
    pass that into a DataGenerator, and wrap that using a PyTorch data
    loader. This is the only function that needs to be used.


    Parameters
    ----------
    peaks: str or pandas.DataFrame or list/tuple of such
            A BED-formatted file containing peak coordinates. This can be either
            the string path to the BED file or a pandas DataFrame object containing
            three columns: chrom, start, and end. Alternatively, this can be a list
            of such objects whose coordinates will be interleaved.

    negatives: str or pandas.DataFrame or list/tuple of such
            A BED-formatted file containing negative coordinates. This can be either
            the string path to the BED file or a pandas DataFrame object containing
            three columns: chrom, start, and end. Alternatively, this can be a list
            of such objects whose coordinates will be interleaved.

    sequences: str or dictionary
            Either the path to a fasta file to read from or a dictionary where the
            keys are the unique set of chromosoms and the values are one-hot
            encoded sequences as numpy arrays or memory maps.

    signals: list of strs or list of dictionaries
            A list of filepaths to bigwig files, where each filepath will be read
            using pyBigWig, or a list of dictionaries where the keys are the same
            set of unique chromosomes and the values are numpy arrays or memory
            maps.

    controls: list of strs or list of dictionaries or None, optional
            A list of filepaths to bigwig files, where each filepath will be read
            using pyBigWig, or a list of dictionaries where the keys are the same
            set of unique chromosomes and the values are numpy arrays or memory
            maps. If None, no control tensor is returned. Default is None.

    chroms: list or None, optional
            A set of chromosomes to extact loci from. Loci in other chromosomes
            in the locus file are ignored. If None, all loci are used. Default is
            None.

    in_window: int, optional
            The input window size. Default is 2114.

    out_window: int, optional
            The output window size. Default is 1000.

    max_jitter: int, optional
            The maximum amount of jitter to add, in either direction, to the
            midpoints that are passed in. Default is 128.

    negative_ratio: float, optional
            The ratio of negatives compared to peaks in each batch. A value of 1 means
            that each batch is balanced, and a value of 10 means that there would be 10
            negatives for each positive. Note that this is independent of the number of
            peaks and negatives provided. Even if the `peaks` input has 10x the number
            of coordinates as the `negatives` one, if the ratio is 1 each batch during
            training will be balanced (on average).

    reverse_complement: bool, optional
            Whether to reverse complement-augment half of the data. Default is True.

    shuffle: bool, optional
            Whether to randomly sample peaks, if True, or to proceed sequentially
            through them, if False. Negatives are always randomly sampled. Default
            is True.

    min_counts: float or None, optional
            The minimum number of counts, summed across the length of each example
            and across all tasks, needed to be kept. If None, no minimum. Default
            is None.

    max_counts: float or None, optional
            The maximum number of counts, summed across the length of each example
            and across all tasks, needed to be kept. If None, no maximum. Default
            is None.

    summits: bool, optional
            Whether to return a region centered around the summit instead of the center
            between the start and end. If True, it will add the 10th column (index 9)
            to the start to get the center of the window, and so the data must be in
            narrowPeak format.

    exclusion_lists: list or None, optional
            A list of strings of filenames to BED-formatted files containing exclusion
            lists, i.e., regions where overlapping loci should be filtered out. If None,
            no filtering is performed based on exclusion zones. Default is None.

    random_state: int or None, optional
            Whether to use a deterministic seed or not.

    pin_memory: bool, optional
            Whether to pin page memory to make data loading onto a GPU easier.
            Default is True.

    num_workers: int, optional
            The number of processes fetching data at a time to feed into a model.
            If 0, data is fetched from the main process. Default is 0.

    batch_size: int, optional
            The number of data elements per batch. Default is 32.

    verbose: bool, optional
            Whether to display a progress bar while loading. Default is False.


    Returns
    -------
    X: torch.utils.data.DataLoader
            A PyTorch DataLoader wrapped DataGenerator object.
    """

    X_peaks = extract_loci(
        loci=peaks,
        sequences=sequences,
        signals=signals,
        in_signals=controls,
        chroms=chroms,
        in_window=in_window,
        out_window=out_window,
        max_jitter=max_jitter,
        min_counts=min_counts,
        max_counts=max_counts,
        summits=summits,
        exclusion_lists=exclusion_lists,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=verbose,
    )

    loci_counts = X_peaks[1].sum(dim=(1, 2))

    outlier_threshold = torch.quantile(X_peaks[1].sum(dim=(1, 2)), 0.99) * 1.2
    outlier_idxs = loci_counts > outlier_threshold

    X_bg = extract_loci(
        loci=negatives,
        sequences=sequences,
        signals=signals,
        in_signals=controls,
        chroms=chroms,
        in_window=in_window,
        out_window=out_window,
        max_jitter=0,
        min_counts=min_counts,
        max_counts=max_counts,
        summits=False,
        exclusion_lists=exclusion_lists,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=verbose,
    )

    if verbose:
        n_filtered_peaks = len(X_peaks[-1]) - X_peaks[-1].sum() + outlier_idxs.sum()
        n_filtered_negatives = len(X_bg[-1]) - X_bg[-1].sum()

        print("\nFiltered Peaks: {}".format(n_filtered_peaks))
        print("Filtered Negatives: {}".format(n_filtered_negatives))

    ###

    X_gen = PeakNegativeSampler(
        peak_sequences=X_peaks[0][~outlier_idxs],
        peak_signals=torch.abs(X_peaks[1][~outlier_idxs]),
        peak_controls=None
        if controls is None
        else torch.abs(X_peaks[2][~outlier_idxs]),
        negative_sequences=X_bg[0],
        negative_signals=torch.abs(X_bg[1]),
        negative_controls=None if controls is None else torch.abs(X_bg[2]),
        negative_ratio=negative_ratio,
        in_window=in_window,
        out_window=out_window,
        max_jitter=max_jitter,
        reverse_complement=reverse_complement,
        shuffle=shuffle,
        random_state=random_state,
    )

    X_gen = torch.utils.data.DataLoader(
        X_gen, pin_memory=pin_memory, num_workers=num_workers, batch_size=batch_size
    )

    return X_gen
