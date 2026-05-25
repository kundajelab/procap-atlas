#!/usr/bin/env python3
"""Generate UCSC track hub files for the ENCODE PRO-cap atlas.

Reads configs/experiment_config.yaml and writes:
  hub/hub.txt
  hub/genomes.txt
  hub/hg38/trackDb.txt

Experiments are grouped into supertracks by biosample, sorted alphabetically.
Each experiment gets a multiWig container for plus/minus strand BigWigs and a
separate bigBed track for peak calls. BPNet attribution BigWigs can also be
exposed as UCSC dynseq tracks.

Usage:
    python src/hub/generate_hub.py --email you@example.com
    python src/hub/generate_hub.py --email you@example.com --output-dir /path/to/hub
    python src/hub/generate_hub.py --email you@example.com --base-url https://example.com/procap-atlas
    python src/hub/generate_hub.py --email you@example.com --track-base-url https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main
    python src/hub/generate_hub.py --email you@example.com --no-attributions
"""

import argparse
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "hub"
DEFAULT_BASE_URL = "https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas"

HUB_TXT_TEMPLATE = """\
hub procap_atlas
shortLabel PRO-cap Atlas
longLabel ENCODE PRO-cap atlas: GRCh38 TSS signal, peaks, and BPNet contribution tracks
genomesFile genomes.txt
email {email}
descriptionUrl {base_url}/
"""

GENOMES_TXT = """\
genome hg38
trackDb hg38/trackDb.txt
"""


def sanitize_track_name(s):
    """Convert a string to a valid UCSC track name (alphanumeric + underscore)."""
    return re.sub(r"[^\w]", "_", s).strip("_")


def make_supertrack_name(biosample):
    return f"supertrack_{sanitize_track_name(biosample).lower()}"


def make_short_label(biosample, exp_id, suffix=""):
    """Make a shortLabel <= 17 chars."""
    acc_suffix = exp_id[5:11]  # e.g. "882DWM" from "ENCSR882DWM"
    label = f"{biosample[:10]} {acc_suffix}{suffix}"
    return label[:17]


def clean_label_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate_label(label, max_len=100):
    """Truncate UCSC longLabel text while preserving a readable ending."""
    label = clean_label_text(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 3].rstrip() + "..."


def experiment_context(exp, exp_id):
    """Return compact experiment context for UCSC longLabel fields."""
    parts = [
        exp.get("biosample_summary", exp.get("biosample", "")),
        exp.get("library_construction", ""),
        f"[{exp_id}]",
    ]
    return " | ".join(clean_label_text(p) for p in parts if clean_label_text(p))


def make_long_label(exp, exp_id, description):
    """Make a descriptive UCSC longLabel <= 100 chars."""
    context = experiment_context(exp, exp_id)
    description = clean_label_text(description)
    if not context:
        return truncate_label(description)
    if not description:
        return truncate_label(context)

    separator = " - "
    available = 100 - len(separator) - len(description)
    if available < 20:
        return truncate_label(f"{context}{separator}{description}")
    return f"{truncate_label(context, available)}{separator}{description}"


def make_supertrack_long_label(biosample):
    return truncate_label(
        f"{biosample} ENCODE PRO-cap collection: observed signal, TSS peaks, and BPNet scores"
    )


def clean_base_url(url):
    return url.rstrip("/") if url else None


def bigwig_url(base_url, track_base_url, processed_path, exp_id, strand):
    if track_base_url:
        return f"{track_base_url}/observed/{exp_id}_{strand}.bigWig"
    filename = Path(processed_path).name
    return f"{base_url}/data/processed/bigwigs/{filename}"


def bigbed_url(base_url, track_base_url, bb_filename):
    if track_base_url:
        return f"{track_base_url}/peaks/bigbed/{bb_filename}"
    return f"{base_url}/data/processed/peaks/{bb_filename}"


def attribution_bigwig_url(base_url, track_base_url, exp_id, head):
    if track_base_url:
        return f"{track_base_url}/attributions/bpnet/{exp_id}_{head}.bigWig"
    return f"{base_url}/attributions/bpnet/bigwigs/{exp_id}_{head}.bigWig"


def write_hub_txt(output_dir, base_url, email):
    (output_dir / "hub.txt").write_text(HUB_TXT_TEMPLATE.format(base_url=base_url, email=email))


def write_genomes_txt(output_dir):
    (output_dir / "genomes.txt").write_text(GENOMES_TXT)


def is_uncapped(exp):
    return "uncapped" in exp.get("library_construction", "").lower()


def write_trackdb(
    output_dir,
    experiments,
    base_url,
    attribution_heads,
    track_base_url=None,
):
    # Group by biosample
    by_biosample = {}
    for exp_id, exp in experiments.items():
        bs = exp.get("biosample", "unknown")
        by_biosample.setdefault(bs, []).append((exp_id, exp))

    lines = []

    for biosample in sorted(by_biosample):
        st_name = make_supertrack_name(biosample)
        lines += [
            f"track {st_name}",
            "superTrack on",
            f"shortLabel {biosample[:17]}",
            f"longLabel {make_supertrack_long_label(biosample)}",
            "",
        ]

        for exp_id, exp in sorted(by_biosample[biosample]):
            processed = exp.get("processed", {})
            pl_path = processed.get("pl_bigwig", "")
            mn_path = processed.get("mn_bigwig", "")
            peaks_path = processed.get("peaks", "")

            biosample_clean = sanitize_track_name(exp.get("biosample", "unknown"))
            bb_filename = f"{exp_id}_{biosample_clean}_peaks.bb"

            signal_name = f"{exp_id}_signal"
            short_signal = make_short_label(biosample, exp_id)
            signal_label = make_long_label(
                exp, exp_id, "observed strand-specific PRO-cap signal"
            )
            visibility = "hide" if is_uncapped(exp) else "full"

            # Composite signal track
            lines += [
                f"track {signal_name}",
                "compositeTrack on",
                f"superTrack {st_name}",
                "type bigWig",
                f"shortLabel {short_signal}",
                f"longLabel {signal_label}",
                f"visibility {visibility}",
                "windowingFunction maximum",
                "maxHeightPixels 128:64:11",
                "priority 1",
                "",
            ]

            # Plus-strand subtrack (positive values, range 0 to 40)
            if pl_path:
                pl_label = make_long_label(
                    exp, exp_id, "plus-strand observed PRO-cap signal"
                )
                lines += [
                    f"    track {exp_id}_pl",
                    f"    parent {signal_name}",
                    f"    bigDataUrl {bigwig_url(base_url, track_base_url, pl_path, exp_id, 'pl')}",
                    f"    shortLabel {(short_signal + ' (+)')[:17]}",
                    f"    longLabel {pl_label}",
                    "    type bigWig 0 40",
                    "    autoScale on",
                    "    color 197,0,11",
                    "    altColor 255,0,0",
                    "    visibility full",
                    "    priority 1",
                    "",
                ]

            # Minus-strand subtrack (values stored negative; range -40 to 0, no negateValues)
            if mn_path:
                mn_label = make_long_label(
                    exp,
                    exp_id,
                    "minus-strand observed PRO-cap signal, negative values",
                )
                lines += [
                    f"    track {exp_id}_mn",
                    f"    parent {signal_name}",
                    f"    bigDataUrl {bigwig_url(base_url, track_base_url, mn_path, exp_id, 'mn')}",
                    f"    shortLabel {(short_signal + ' (-)')[:17]}",
                    f"    longLabel {mn_label}",
                    "    type bigWig -40 0",
                    "    autoScale on",
                    "    color 0,132,209",
                    "    altColor 0,0,255",
                    "    visibility full",
                    "    priority 2",
                    "",
                ]

            # Peaks track
            if peaks_path:
                peak_label = make_long_label(
                    exp, exp_id, "merged bidirectional/unidirectional TSS peaks"
                )
                lines += [
                    f"track {exp_id}_peaks",
                    f"superTrack {st_name}",
                    "type bigBed 3 +",
                    f"bigDataUrl {bigbed_url(base_url, track_base_url, bb_filename)}",
                    f"shortLabel {make_short_label(biosample, exp_id, ' pk')}",
                    f"longLabel {peak_label}",
                    f"visibility {'hide' if is_uncapped(exp) else 'dense'}",
                    "priority 2",
                    "",
                ]

            # BPNet contribution-score dynseq tracks. Keep these as standalone
            # bigWig tracks because UCSC's dynseq logo display is track-level.
            if attribution_heads:
                for priority, head in enumerate(attribution_heads, start=1):
                    head_label = "Profile" if head == "profile" else "Count"
                    attr_label = make_long_label(
                        exp,
                        exp_id,
                        f"BPNet {head_label.lower()} contribution scores",
                    )
                    if is_uncapped(exp):
                        default_visibility = "hide"
                    else:
                        default_visibility = "full"
                    lines += [
                        f"track {exp_id}_attr_{head}",
                        f"superTrack {st_name}",
                        f"bigDataUrl {attribution_bigwig_url(base_url, track_base_url, exp_id, head)}",
                        f"shortLabel {(short_signal + ' ' + head[:4])[:17]}",
                        f"longLabel {attr_label}",
                        "type bigWig",
                        "logo on",
                        "autoScale on",
                        "alwaysZero on",
                        "mouseOverFunction noAverage",
                        "maxHeightPixels 128:64:16",
                        f"visibility {default_visibility}",
                        f"priority {2 + priority}",
                        "",
                    ]

    trackdb_path = output_dir / "hg38" / "trackDb.txt"
    trackdb_path.parent.mkdir(parents=True, exist_ok=True)
    trackdb_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate UCSC track hub for the ENCODE PRO-cap atlas"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, metavar="PATH")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, metavar="URL")
    parser.add_argument(
        "--track-base-url",
        default=None,
        metavar="URL",
        help="base URL for observed/attribution BigWigs, e.g. a Hugging Face dataset resolve URL",
    )
    parser.add_argument("--email", required=True, metavar="EMAIL")
    parser.add_argument(
        "--attribution-head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="BPNet attribution head(s) to include; repeatable (default: profile count)",
    )
    parser.add_argument(
        "--no-attributions",
        action="store_true",
        help="omit BPNet attribution dynseq tracks",
    )
    args = parser.parse_args()

    attribution_heads = [] if args.no_attributions else args.attribution_head
    if attribution_heads is None:
        attribution_heads = ["profile", "count"]
    track_base_url = clean_base_url(args.track_base_url)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_hub_txt(args.output_dir, args.base_url, args.email)
    write_genomes_txt(args.output_dir)
    write_trackdb(
        args.output_dir,
        experiments,
        args.base_url,
        attribution_heads,
        track_base_url=track_base_url,
    )

    n = len(experiments)
    print(f"Wrote hub for {n} experiments to {args.output_dir}/")
    print(f"  {args.output_dir}/hub.txt")
    print(f"  {args.output_dir}/genomes.txt")
    print(f"  {args.output_dir}/hg38/trackDb.txt")


if __name__ == "__main__":
    main()
