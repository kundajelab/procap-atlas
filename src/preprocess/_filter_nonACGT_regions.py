"""
Filter non-ACGT regions from a BED file.

For each region in the input BED file, extracts a sequence of length
--in_window (default 2114) centered on the region midpoint and removes it if:
  - the chromosome is not present in the reference FASTA, or
  - the extracted sequence contains any non-ACGT characters (e.g. N), or
  - the extracted sequence is shorter than --in_window (e.g. near a contig edge).

Usage:
    python _filter_nonACGT_regions.py -b peaks.bed -f hg38.fa -o peaks.filtered.bed
    python _filter_nonACGT_regions.py -b peaks.bed -f hg38.fa -o peaks.filtered.bed -w 2114 -v
"""

import argparse

import numpy as np
import pandas as pd
import pyfaidx
import tqdm


def filter_nonACGT_regions(bed_fp, fa_fp, in_window=2114, verbose=False):
    """
    Filter out BED regions whose centered sequence contains non-ACGT characters,
    is shorter than in_window (e.g. near a contig edge), or lies on a chromosome
    not present in the reference FASTA.

    Parameters
    ----------
    bed_fp : str
        Path to input BED file.
    fa_fp : str
        Path to reference FASTA file (indexed by pyfaidx).
    in_window : int, optional
        Sequence length to extract centered on each region midpoint. Default 2114.
    verbose : bool, optional
        Whether to show a progress bar. Default False.

    Returns
    -------
    pd.DataFrame
        Filtered BED as a DataFrame (rows passing all checks).
    """
    snp_bed = pd.read_csv(bed_fp, sep="\t", header=None)
    fa = pyfaidx.Fasta(fa_fp)
    chroms = set(fa.keys())
    wholesome = []
    for row in tqdm.tqdm(
        snp_bed.itertuples(), total=snp_bed.shape[0], disable=not verbose
    ):
        chrom = str(row[1])
        center = (row[2] + row[3]) // 2
        start = max(0, center - in_window // 2)
        end = start + in_window
        if chrom in chroms:
            seq = str(fa[chrom][start:end]).upper()
            is_wholesome = all([c in "ACGT" for c in seq]) and len(seq) == in_window
        else:
            is_wholesome = False
        wholesome.append(is_wholesome)

    print(
        f"Filtered out {sum(~np.array(wholesome))} due to non-ACGT characters, "
        f"length != {in_window}, or invalid chromosome."
    )
    return snp_bed[wholesome]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--bed_fp", type=str, required=True)
    parser.add_argument("-f", "--fa_fp", type=str, required=True)
    parser.add_argument("-o", "--out_fp", type=str, required=True)
    parser.add_argument("-w", "--in_window", type=int, default=2114)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    filter_bed = filter_nonACGT_regions(
        args.bed_fp, args.fa_fp, args.in_window, args.verbose
    )
    filter_bed.to_csv(args.out_fp, sep="\t", index=False, header=False)
