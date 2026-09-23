#!/usr/bin/env python3
"""Motif-centered PRO-cap metaplots: observed BigWig signal aligned to
instances of one motif, pulled from either Fi-NeMo hit calls or raw
TF-MoDISco/MotifCompendium seqlets.

Three sources, chosen with `--source`:

  hits                one experiment's own Fi-NeMo hits.tsv/hits_unique.tsv
                       (or hits_linked.tsv with --use-compendium-name), same
                       convention as diagnose_hit_signal_metaplot.py.
  seqlets              one experiment's own per-experiment TF-MoDISco h5
                       (modisco/bpnet/{experiment}_{head}.modisco.h5),
                       *before* any Fi-NeMo hit-calling -- useful for
                       checking a motif that Fi-NeMo hasn't been run for yet,
                       or for comparing seqlet-level vs. hit-level signal
                       directly.
  compendium-seqlets   every experiment contributing to one MotifCompendium
                       cluster, via motifcompendium_{head}_pattern_to_cluster.tsv
                       -- pools raw discovery-level signal atlas-wide for one
                       cluster_final id, independent of whether Fi-NeMo has
                       been (re-)run against that cluster for every
                       experiment yet.

Resolving a seqlet to a genome position needs peaks.narrowPeak's own peak
coordinates -- *not* regions.npz. regions.npz packages the same peak
coordinates together with a cropped copy of the sequence/contribution
arrays (via finemo's `extract-regions-modisco-fmt`), but this module has no
use for those arrays, so building it is unnecessary work, and it depends on
the head-specific attributions .npz for something this module never reads
(regions.npz's build fails if that file is missing/corrupt, even though
only the coordinates -- derived solely from peaks.narrowPeak and the shared
{experiment}_ohe.npz -- are actually needed here; see finemo's own
`load_peaks()` in data_io.py). peaks.narrowPeak is also written earlier and
independently of that attribution step (call_hits_bpnet.py's
build_peaks_narrowpeak, before its `finemo extract-regions-modisco-fmt`
call) and is protected from `cleanup_hitcalls.py`'s deletions, so it stays
available even when regions.npz has been cleaned up or never finished
building.

Getting the peak coordinate right isn't the whole story, though: reading
peaks.narrowPeak at half-width = (this experiment's OHE array width)//2
gives peak_region_start in the *raw* attribution window's frame (IN_WINDOW,
2114bp by default) -- but a seqlet's start/end are local to a SECOND,
narrower crop modisco-lite's own CLI takes before seqlet discovery
(`modisco motifs -w/--window`, 1000bp in this repo's modisco.sh), centered
on the same peak. Missing that second crop offset (as this module did until
2026-09-23) silently shifts every resolved position by a fixed amount
((raw_width - modisco_window) // 2 -- 557bp for 2114/1000) in every
instance alike, which isn't obviously wrong the way a crash or a scattered
random shift would be: it just averages out to flat, no-signal-looking
metaplots, since every instance is displaced from the true motif position
by the same fixed offset rather than centered on it. `seqlet_positions()`
reads the exact window size modisco-lite used back from the h5's own
`window_size` attribute (written by `modiscolite.io.save_hdf5`) rather than
assuming/hardcoding it. Fi-NeMo hits need no such correction at all: their
genome coordinates are written directly into hits.tsv by call-hits.

Seqlet/hit "strand" is a locally-discovered pattern's own orientation label
and has no guaranteed relationship to real transcription direction (a motif
can be labelled "+" from one experiment's MoDISco run and get its reverse
complement labelled "+" in another) -- collect_windows() is fed this label
directly to orient windows, with no further correction. An earlier version
of this module (like diagnose_hit_signal_metaplot.py, which this copied the
idea from) additionally auto-detected orientation from the signal itself --
flip if antisense exceeds sense, whether checked at the center bin or
summed over the whole window -- and applied that per contributing
experiment before pooling. That's circular (deciding orientation from the
outcome you're measuring can manufacture a peak-looking shape out of
noise) and, worse, noisy at that granularity: a per-experiment decision
made from that one experiment's own often-small seqlet set has nothing
independent to calibrate against, unlike diagnose_hit_signal_metaplot.py's
one-time decision from a single large, trusted reference group. Removed
entirely rather than re-tuned. If a compendium cluster's pooled signal ever
looks orientation-inverted, that means Fi-NeMo/MoDISco's own strand label
is wrong for enough contributing experiments to matter, which is a
labelling bug to fix upstream, not something to paper over here.

Usage:
    python src/bpnet/hitcall/metaplot_motif.py --source hits \\
        -e ENCSR342WAR --head profile --motif-name pos_patterns.pattern_2
    python src/bpnet/hitcall/metaplot_motif.py --source hits \\
        -e ENCSR342WAR --head count --motif-name pos_patterns.42 --use-compendium-name
    python src/bpnet/hitcall/metaplot_motif.py --source seqlets \\
        -e ENCSR342WAR --head profile --motif-name pos_patterns.pattern_2
    python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \\
        --head count --compendium-motif-name pos_patterns.42
"""

import argparse
import re
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.format as npy_format
import pandas as pd
import yaml
from finemo.data_io import load_peaks

import compressed_io
from call_hits_bpnet import resolve_experiment_paths, resolve_hits_path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "metaplot"))
from metaplot_tss import collect_windows  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
ATTR_DIR = REPO_ROOT / "attributions" / "bpnet"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
DEFAULT_WINDOW = 200
DEFAULT_BIN_SIZE = 5
LOCAL_MOTIF_RE = re.compile(r"^(pos|neg)_patterns\.(pattern_\d+)$")


def npz_array_shape(path: str, key: str = "arr_0") -> tuple[int, ...]:
    """One array's shape inside a .npz, without decompressing its data.

    region_geometry() only ever wants {experiment}_ohe.npz's last
    dimension, but a plain np.load()/load_npy_or_npz() must fully
    decompress the member to hand back an ndarray -- it's a zip archive,
    not mmap-able. compendium-seqlets mode calls this once per contributing
    experiment, and a widely-shared ("ubiquitous") cluster's contributing
    experiments can number in the hundreds, each with a multi-GB
    uncompressed OHE array -- decompressing every one of those just to
    read one int was the dominant cost in a real timing (0.75s vs. 0.0003s
    for a 50k-peak array in a synthetic benchmark). The .npy format's shape
    lives in a small text header at the very start of the member's
    decompressed stream, so this reads only that header -- deflate
    decoding is sequential, so reading the first few hundred bytes of
    output only costs decompressing roughly that much input, not the
    array's full data.
    """
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
        major, _ = npy_format.read_magic(f)
        read_header = (
            npy_format.read_array_header_1_0 if major == 1
            else npy_format.read_array_header_2_0
        )
        shape, _, _ = read_header(f)
        return shape


_READ_COUNTS: dict[str, float] | None = None


def load_total_reads(experiment: str) -> float | None:
    """configs/n_reads.txt's total-reads column, parsed once and cached.

    compendium-seqlets mode looks this up once per contributing experiment
    per cluster, and the same experiment recurs across many clusters within
    one process -- a per-call linear scan of the file re-read it from
    scratch every time.
    """
    global _READ_COUNTS
    if _READ_COUNTS is None:
        _READ_COUNTS = {}
        with open(N_READS_PATH) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    _READ_COUNTS[parts[0]] = float(parts[4])
    return _READ_COUNTS.get(experiment)


def bigwig_paths(config: dict, experiment: str) -> tuple[Path, Path]:
    processed = config["experiments"][experiment]["processed"]
    return REPO_ROOT / processed["pl_bigwig"], REPO_ROOT / processed["mn_bigwig"]


def parse_local_motif_name(name: str) -> tuple[str, str]:
    """"pos_patterns.pattern_3" -> ("pos_patterns", "pattern_3")."""
    m = LOCAL_MOTIF_RE.match(name)
    if not m:
        raise SystemExit(
            f"Error: {name!r} doesn't look like a local motif name "
            "(expected e.g. 'pos_patterns.pattern_3')"
        )
    return f"{m.group(1)}_patterns", m.group(2)


def hit_positions(
    experiment: str, head: str, motif_name: str, min_trim_len: int | None,
    model_dir: str | None, use_compendium_name: bool, verbose: bool,
) -> list[tuple[str, int, str]]:
    """(chr, center, strand) for every Fi-NeMo hit of `motif_name`."""
    _, hits_dir, _, _ = resolve_experiment_paths(experiment, head, min_trim_len, model_dir)
    if use_compendium_name:
        hits_path = hits_dir / "hits_linked.tsv"
        if not compressed_io.exists(hits_path):
            raise SystemExit(
                f"Error: {hits_path} missing -- run link_hits_to_compendium.py first"
            )
        hits_path = compressed_io.resolve(hits_path)
        column = "compendium_motif_name"
    else:
        hits_path = resolve_hits_path(hits_dir, verbose=verbose)
        if hits_path is None:
            raise SystemExit(f"Error: no hits found in {hits_dir}")
        column = "motif_name"

    if verbose:
        print(f"Reading hits from {hits_path}")
    hits = pd.read_csv(hits_path, sep="\t")
    hits = hits[hits[column] == motif_name]
    return [
        (row.chr, int((row.start + row.end) // 2), row.strand)
        for row in hits.itertuples(index=False)
    ]


@lru_cache(maxsize=None)
def region_geometry(
    experiment: str, head: str, min_trim_len: int | None, model_dir: str | None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """(peaks_df, raw_width). peaks_df ('chr'/'peak_region_start') is
    row-aligned with the OHE/attribution arrays TF-MoDISco seqlets index
    into -- read directly from peaks.narrowPeak rather than regions.npz. See
    module docstring for why. raw_width is returned too because
    seqlet_positions() needs it a second time, to correct for modisco's own
    -w/--window crop (see module docstring) -- returning it here avoids
    decompressing {experiment}_ohe.npz a second time for the same value.

    Cached per (experiment, head, min_trim_len, model_dir): both inputs are
    files on disk that don't change within one process's run, but
    compendium-seqlets mode calls this once per contributing experiment for
    *every* cluster, and a "ubiquitous" cluster's contributing experiments
    heavily overlap the next one's. Reading raw_width is cheap now
    (npz_array_shape() reads only the .npy header, not the array), but
    resolving peaks.narrowPeak still means a bgzip decompression and a
    polars parse per call, which this still saves across those repeats.
    """
    exp_dir, _, _, _ = resolve_experiment_paths(experiment, head, min_trim_len, model_dir)
    peaks_narrowpeak = compressed_io.resolve(exp_dir / "peaks.narrowPeak", missing_ok=True)
    if peaks_narrowpeak is None:
        raise SystemExit(
            f"Error: peaks.narrowPeak missing in {exp_dir} -- run "
            "extract_regions_bpnet.py first"
        )

    ohe_path = ATTR_DIR / f"{experiment}_ohe.npz"
    if not ohe_path.exists():
        raise SystemExit(f"Error: {ohe_path} missing")
    raw_width = npz_array_shape(str(ohe_path))[-1]

    if verbose:
        print(f"Reading peak coordinates from {peaks_narrowpeak}")
    peaks_df = load_peaks(str(peaks_narrowpeak), None, raw_width // 2)
    return peaks_df.to_pandas(), raw_width


def seqlet_positions(
    experiment: str, head: str, local_motif_name: str, min_trim_len: int | None,
    model_dir: str | None, modisco_h5: Path | None, verbose: bool,
) -> list[tuple[str, int, int, bool]]:
    """(chr, genome_start, genome_end, is_revcomp) for every seqlet TF-MoDISco
    assigned to `local_motif_name` in this experiment's own per-experiment
    MoDISco h5.
    """
    h5_path = modisco_h5 or (MODISCO_DIR / f"{experiment}_{head}.modisco.h5")
    if not Path(h5_path).exists():
        raise SystemExit(f"Error: {h5_path} missing")
    posneg_group, pattern_key = parse_local_motif_name(local_motif_name)

    peaks_df, raw_width = region_geometry(experiment, head, min_trim_len, model_dir, verbose)
    chrs = peaks_df["chr"].to_numpy()
    region_starts = peaks_df["peak_region_start"].to_numpy()

    if verbose:
        print(f"Reading seqlets from {h5_path}:{posneg_group}/{pattern_key}")
    with h5py.File(h5_path, "r") as f:
        if posneg_group not in f or pattern_key not in f[posneg_group]:
            raise SystemExit(
                f"Error: {posneg_group}/{pattern_key} not found in {h5_path}"
            )
        if "window_size" not in f.attrs:
            raise SystemExit(
                f"Error: {h5_path} has no window_size attribute -- written "
                "by an older modisco-lite? seqlet coordinates can't be "
                "resolved without it (see module docstring)."
            )
        # modisco-lite's own `motifs -w/--window` crops the raw attribution
        # array to a window centered on the peak *before* seqlet discovery
        # (modisco.sh uses -w 1000, well under IN_WINDOW's 2114) -- every
        # seqlet's start/end is local to THAT crop, not the full raw array.
        # window_size is the exact value used, saved as an h5 attribute by
        # modiscolite.io.save_hdf5, so this is read back rather than
        # assumed/hardcoded. Missing this offset (as this module did until
        # 2026-09-23) silently shifts every resolved position by
        # (raw_width - window_size) // 2 -- 557bp for the atlas's own
        # 2114/1000 combination -- which averages out to flat/no signal in
        # a metaplot instead of erroring, since it's a systematic shift
        # applied identically to every seqlet, not random.
        modisco_window = int(f.attrs["window_size"])
        seqlets = f[posneg_group][pattern_key]["seqlets"]
        starts = seqlets["start"][:]
        ends = seqlets["end"][:]
        example_idx = seqlets["example_idx"][:]
        is_revcomp = seqlets["is_revcomp"][:]

    crop_start = raw_width // 2 - modisco_window // 2

    out = []
    n_regions = len(chrs)
    for s, e, idx, rc in zip(starts, ends, example_idx, is_revcomp):
        idx = int(idx)
        if idx < 0 or idx >= n_regions:
            continue
        genome_start = int(region_starts[idx]) + crop_start + int(s)
        genome_end = int(region_starts[idx]) + crop_start + int(e)
        out.append((str(chrs[idx]), genome_start, genome_end, bool(rc)))
    return out


def compendium_experiments(
    compendium_motif_name: str, head: str, mapping_tsv: Path | None,
) -> list[tuple[str, str]]:
    """(experiment, local_motif_name) pairs contributing to one compendium
    cluster, from motifcompendium_{head}_pattern_to_cluster.tsv.
    """
    path = mapping_tsv or (MC_DIR / f"motifcompendium_{head}_pattern_to_cluster.tsv")
    if not compressed_io.exists(path):
        raise SystemExit(
            f"Error: {path} missing -- run motifcompendium/cluster_motifs.py first"
        )
    mapping = pd.read_csv(compressed_io.resolve(path), sep="\t")
    sub = mapping[mapping["compendium_motif_name"] == compendium_motif_name]
    if sub.empty:
        raise SystemExit(
            f"Error: {compendium_motif_name!r} not found in {path}"
        )
    return list(zip(sub["experiment"], sub["local_motif_name"]))


def collect_metaplot(
    source: str, head: str, config: dict, *,
    experiment: str | None = None, motif_name: str | None = None,
    compendium_motif_name: str | None = None, mapping_tsv: Path | None = None,
    min_trim_len: int | None = None, model_dir: str | None = None,
    use_compendium_name: bool = False, modisco_h5: Path | None = None,
    window: int = DEFAULT_WINDOW, bin_size: int = DEFAULT_BIN_SIZE,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """(mean_sense, mean_antisense, n_instances) for one motif, from
    whichever source -- the shared logic behind all three of main()'s
    branches, factored out so other scripts (select_motif_exemplars.py,
    plot_figure2.py) can call this directly instead of shelling out to this
    script as a subprocess per motif. Raises SystemExit on any input problem,
    same as the CLI.
    """
    sense_rows, antisense_rows = [], []

    if source == "hits":
        if not (experiment and motif_name):
            raise SystemExit("collect_metaplot(source='hits') needs experiment and motif_name")
        tss_list = hit_positions(
            experiment, head, motif_name, min_trim_len, model_dir,
            use_compendium_name, verbose,
        )
        total_reads = load_total_reads(experiment)
        if total_reads is None:
            raise SystemExit(f"Error: {experiment} not in {N_READS_PATH}")
        pl_path, mn_path = bigwig_paths(config, experiment)
        sense, antisense = collect_windows(
            pl_path, mn_path, tss_list, window, bin_size, total_reads
        )
        if verbose:
            print(f"{experiment}: {len(tss_list)} hits, {len(sense)} windows extracted")
        if len(sense) == 0:
            raise SystemExit("Error: no windows extracted")
        sense_rows.append(sense.mean(axis=0))
        antisense_rows.append(antisense.mean(axis=0))
        n_total = len(sense)

    elif source == "seqlets":
        if not (experiment and motif_name):
            raise SystemExit("collect_metaplot(source='seqlets') needs experiment and motif_name")
        positions = seqlet_positions(
            experiment, head, motif_name, min_trim_len, model_dir, modisco_h5, verbose,
        )
        tss_list = [
            (chrom, (start + end) // 2, "-" if rc else "+")
            for chrom, start, end, rc in positions
        ]
        total_reads = load_total_reads(experiment)
        if total_reads is None:
            raise SystemExit(f"Error: {experiment} not in {N_READS_PATH}")
        pl_path, mn_path = bigwig_paths(config, experiment)
        sense, antisense = collect_windows(
            pl_path, mn_path, tss_list, window, bin_size, total_reads
        )
        if verbose:
            print(f"{experiment}: {len(positions)} seqlets, {len(sense)} windows extracted")
        if len(sense) == 0:
            raise SystemExit("Error: no windows extracted")
        sense_rows.append(sense.mean(axis=0))
        antisense_rows.append(antisense.mean(axis=0))
        n_total = len(sense)

    elif source == "compendium-seqlets":
        if not compendium_motif_name:
            raise SystemExit(
                "collect_metaplot(source='compendium-seqlets') needs compendium_motif_name"
            )
        pairs = compendium_experiments(compendium_motif_name, head, mapping_tsv)
        if verbose:
            print(f"{compendium_motif_name}: {len(pairs)} contributing experiment(s)")
        n_total = 0
        for exp, local_motif_name in pairs:
            try:
                positions = seqlet_positions(
                    exp, head, local_motif_name, min_trim_len, model_dir, None, verbose,
                )
                total_reads = load_total_reads(exp)
                if total_reads is None:
                    if verbose:
                        print(f"  {exp}: WARNING no read count, skipping", file=sys.stderr)
                    continue
                pl_path, mn_path = bigwig_paths(config, exp)
            except SystemExit as e:
                if verbose:
                    print(f"  {exp}: WARNING {e}, skipping", file=sys.stderr)
                continue

            tss_list = [
                (chrom, (start + end) // 2, "-" if rc else "+")
                for chrom, start, end, rc in positions
            ]
            sense, antisense = collect_windows(
                pl_path, mn_path, tss_list, window, bin_size, total_reads
            )
            if len(sense) == 0:
                if verbose:
                    print(f"  {exp}: 0 windows extracted, skipping", file=sys.stderr)
                continue
            if verbose:
                print(f"  {exp}: {len(positions)} seqlets, {len(sense)} windows")
            sense_rows.append(sense.mean(axis=0))
            antisense_rows.append(antisense.mean(axis=0))
            n_total += len(sense)

        if not sense_rows:
            raise SystemExit("Error: no windows extracted from any contributing experiment")

    else:
        raise SystemExit(f"Error: unknown source {source!r}")

    return np.mean(sense_rows, axis=0), np.mean(antisense_rows, axis=0), n_total


def draw_metaplot(
    ax, sense: np.ndarray, antisense: np.ndarray, window: int, bin_size: int,
) -> None:
    """Motif-centered sense/antisense signal on a caller-supplied axes, so
    this can be dropped into a shared grid (e.g. beside a motif's own CWM
    logo) instead of always owning its own figure.
    """
    n_bins = 2 * window // bin_size
    positions = np.linspace(-window, window, n_bins, endpoint=False) + bin_size / 2
    ax.fill_between(positions, sense, color="#e04b4b", alpha=0.8, label="Sense")
    ax.fill_between(positions, -antisense, color="#4b7be0", alpha=0.8, label="Antisense")
    ax.plot(positions, sense, color="#c02020", linewidth=0.8)
    ax.plot(positions, -antisense, color="#2050c0", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ymax = max(float(sense.max()), float(antisense.max()), 1e-6) * 1.15
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-window, window)


def plot_motif_metaplot(
    sense: np.ndarray, antisense: np.ndarray, window: int, bin_size: int,
    n: int, title: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    draw_metaplot(ax, sense, antisense, window, bin_size)
    ax.set_xlabel("Position relative to motif center (bp)")
    ax.set_ylabel("Mean signal (RPM)")
    ax.set_title(f"{title}\nn = {n:,} instances")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, choices=["hits", "seqlets", "compendium-seqlets"],
    )
    parser.add_argument("-e", "--experiment", type=str, default=None,
                        help="required for --source hits/seqlets")
    parser.add_argument("-m", "--model-dir", type=str, default=None)
    parser.add_argument("--head", type=str, default="profile", choices=["profile", "count"])
    parser.add_argument("--min-trim-len", type=int, default=None, metavar="BP")
    parser.add_argument(
        "--motif-name", type=str, default=None,
        help="local motif name for --source hits/seqlets, e.g. pos_patterns.pattern_2 "
        "(or a compendium name with --source hits --use-compendium-name)",
    )
    parser.add_argument(
        "--use-compendium-name", action="store_true",
        help="--source hits only: filter hits_linked.tsv's compendium_motif_name "
        "instead of the local motif_name (needs link_hits_to_compendium.py run first)",
    )
    parser.add_argument(
        "--compendium-motif-name", type=str, default=None,
        help="required for --source compendium-seqlets, e.g. pos_patterns.42",
    )
    parser.add_argument("--mapping-tsv", type=Path, default=None,
                        help="override motifcompendium_{head}_pattern_to_cluster.tsv")
    parser.add_argument("--modisco-h5", type=Path, default=None,
                        help="--source seqlets only: override the per-experiment modisco h5")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, metavar="BP")
    parser.add_argument("--bin-size", type=int, default=DEFAULT_BIN_SIZE, metavar="BP")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures" / "metaplots")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.source in ("hits", "seqlets") and not args.experiment:
        parser.error(f"--source {args.source} requires -e/--experiment")
    if args.source in ("hits", "seqlets") and not args.motif_name:
        parser.error(f"--source {args.source} requires --motif-name")
    if args.source == "compendium-seqlets" and not args.compendium_motif_name:
        parser.error("--source compendium-seqlets requires --compendium-motif-name")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    label = args.motif_name or args.compendium_motif_name
    mean_sense, mean_antisense, n_total = collect_metaplot(
        args.source, args.head, config,
        experiment=args.experiment, motif_name=args.motif_name,
        compendium_motif_name=args.compendium_motif_name,
        mapping_tsv=args.mapping_tsv, min_trim_len=args.min_trim_len,
        model_dir=args.model_dir, use_compendium_name=args.use_compendium_name,
        modisco_h5=args.modisco_h5, window=args.window, bin_size=args.bin_size,
        verbose=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(".", "_")
    out_path = args.out_dir / f"{args.source}_{args.head}_{safe_label}_metaplot.png"
    plot_motif_metaplot(
        mean_sense, mean_antisense, args.window, args.bin_size, n_total,
        f"{args.source}: {label} ({args.head} head)", out_path,
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
