#!/usr/bin/env python3
"""Build a hand-annotation scaffold for compendium clusters JASPAR can't label.

JASPAR misses whole categories that matter here: core promoter elements
(Inr, TATA, DPE), repeats, and composite arrangements. On the real count head
37 of the 343 reproducible clusters carry no JASPAR name, and reviewing them by
eye showed the class is largely tandem composites -- a double CCAAT box, a
double GGAAT -- whose cores *are* in JASPAR but not as repeats, so the
nearest-neighbour lookup fails. Those need a human label, which the manuscript
methods already commit to providing.

Emits two files keyed on the same `cluster_final`:

  ..._scaffold.tsv    one row per cluster with a blank `class` column to fill,
                      plus prevalence, seqlets, tissue restriction and the
                      logo path, sorted by seqlets so the consequential ones
                      come first.
  ..._scaffold.html   the same rows with the logo embedded, to look at while
                      filling in the TSV.

The completed TSV is what `plot_motif_rarefaction.py --annotation-tsv` and
`motif_group_concentration.py --annotation-tsv` consume, so stratified curves
and per-class concentration follow directly from the annotation with no
further work.

Defaults to the clusters with no JASPAR name, since those are the ones needing
a label; `--all` includes every cluster, and `--min-prevalence` restricts to
the reproducible ones.

Usage:
    python src/analysis/make_annotation_scaffold.py --head count
    python src/analysis/make_annotation_scaffold.py --head count --min-prevalence 2
    python src/analysis/make_annotation_scaffold.py --head count --all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_redundancy import embed_svg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"

# Suggested vocabulary, printed as a comment in the TSV. Deliberately short:
# these feed --annotation-tsv's stratified curves, which are only readable with
# a handful of classes.
SUGGESTED_CLASSES = (
    "core_promoter",      # Inr, TATA, DPE, BREu/d
    "tandem_composite",   # same core repeated in one window (double CCAAT)
    "hetero_composite",   # two different motifs in one window
    "repeat",             # low-complexity / repeat-derived
    "tf_unannotated",     # looks like a real TF motif JASPAR lacks
    "unclear",            # not judgeable from the logo
)


def load_clusters(
    metadata: Path,
    concentration: Path | None,
    logo_paths: Path | None,
    logo_root: Path,
) -> pd.DataFrame:
    """Join cluster metadata with tissue restriction and logo paths."""
    meta = pd.read_csv(metadata, sep="\t")
    missing = {"cluster_final", "posneg"} - set(meta.columns)
    if missing:
        raise ValueError(f"{metadata} is missing {sorted(missing)}")
    meta["motif"] = (
        meta["posneg"].astype(str) + "_patterns."
        + meta["cluster_final"].astype(int).astype(str)
    )
    if {"total_seqlets", "n_motifs"} <= set(meta.columns):
        meta["seqlets_per_motif"] = (
            meta["total_seqlets"] / meta["n_motifs"].replace(0, pd.NA)
        ).round(1)

    if concentration is not None and Path(concentration).exists():
        conc = pd.read_csv(concentration, sep="\t")
        keep = [
            c for c in ("cluster_final", "prevalence", "n_groups", "sole_group",
                        "concentration", "is_single_group")
            if c in conc.columns
        ]
        meta = meta.merge(conc[keep], on="cluster_final", how="left")

    if logo_paths is not None and Path(logo_paths).exists():
        lp = pd.read_csv(logo_paths, sep="\t")
        if {"cluster_final", "logo_fwd_svg"} <= set(lp.columns):
            rel = {int(k): v for k, v in zip(lp["cluster_final"], lp["logo_fwd_svg"])}
            meta["logo"] = meta["cluster_final"].map(
                lambda c: str((logo_root / rel[int(c)]).resolve())
                if int(c) in rel else None
            )
    return meta


def write_scaffold_html(rows: pd.DataFrame, path: Path, head: str) -> None:
    """Logos plus the row data, for filling the TSV against."""
    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;margin:18px;font-size:13px}",
        "table{border-collapse:collapse}td,th{padding:5px 9px;",
        "border-bottom:1px solid #ddd;vertical-align:middle}",
        "img{height:64px}code{font-size:11px;color:#444}",
        "</style></head><body>",
        f"<h2>Annotation scaffold — {head} head, {len(rows)} clusters</h2>",
        "<p>Fill the <code>class</code> column of the matching "
        "<code>_scaffold.tsv</code>, keyed on <code>cluster_final</code>. "
        "Suggested values: <code>" + "</code>, <code>".join(SUGGESTED_CLASSES)
        + "</code>. Sorted by seqlet count, so the clusters that most affect "
        "the lexicon come first.</p>",
        "<table><tr><th>cluster_final</th><th>logo</th><th>prevalence</th>"
        "<th>seqlets</th><th>seqlets/motif</th><th>tissue</th></tr>",
    ]
    for _, r in rows.iterrows():
        logo = r.get("logo")
        uri = embed_svg(Path(str(logo))) if logo and not pd.isna(logo) else None
        img = f"<img src='{uri}'>" if uri else "&mdash;"
        tissue = r.get("sole_group") or ""
        if pd.isna(tissue):
            tissue = ""
        html.append(
            f"<tr><td><code>{int(r['cluster_final'])}</code></td><td>{img}</td>"
            f"<td>{r.get('prevalence', '')}</td><td>{r.get('total_seqlets', '')}</td>"
            f"<td>{r.get('seqlets_per_motif', '')}</td><td>{tissue}</td></tr>"
        )
    html.append("</table></body></html>")
    path.write_text("\n".join(html))
    print(f"Saved {path}  ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", default="count", choices=["profile", "count"])
    parser.add_argument("--cluster-metadata", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--concentration-tsv", type=Path, default=None, metavar="PATH",
        help="motif_concentration_{head}_tissue.tsv, for prevalence and tissue "
             "restriction columns (default: auto-detect under figures/)",
    )
    parser.add_argument("--logo-paths", type=Path, default=None, metavar="PATH")
    parser.add_argument("--logo-root", type=Path, default=None, metavar="DIR")
    parser.add_argument(
        "--all", action="store_true",
        help="include every cluster, not just those with no JASPAR name",
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=0, metavar="N",
        help="only clusters seen in >= N experiments (default: 0)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR",
    )
    args = parser.parse_args()

    metadata = args.cluster_metadata or (
        MC_DIR / f"motifcompendium_{args.head}_cluster_metadata.tsv"
    )
    if not metadata.exists():
        print(f"ERROR: cluster metadata not found: {metadata}", file=sys.stderr)
        sys.exit(1)
    conc = args.concentration_tsv
    if conc is None:
        candidate = args.out_dir / f"motif_concentration_{args.head}_tissue.tsv"
        conc = candidate if candidate.exists() else None
    logos = args.logo_paths
    if logos is None:
        candidate = MC_DIR / f"motifcompendium_{args.head}_cluster_logo_paths.tsv"
        logos = candidate if candidate.exists() else None

    meta = load_clusters(metadata, conc, logos, args.logo_root or MC_DIR)
    print(f"{args.head}: {len(meta)} clusters in {metadata.name}", file=sys.stderr)

    rows = meta
    if not args.all:
        if "jaspar_name" not in rows.columns:
            print("ERROR: no jaspar_name column; use --all", file=sys.stderr)
            sys.exit(1)
        unnamed = rows["jaspar_name"].isna() | (rows["jaspar_name"].astype(str) == "")
        rows = rows[unnamed]
        print(f"  {len(rows)} with no JASPAR name", file=sys.stderr)
    if args.min_prevalence > 0:
        if "prevalence" not in rows.columns:
            print(
                "ERROR: --min-prevalence needs a prevalence column; pass "
                "--concentration-tsv",
                file=sys.stderr,
            )
            sys.exit(1)
        rows = rows[rows["prevalence"] >= args.min_prevalence]
        print(f"  {len(rows)} at prevalence >= {args.min_prevalence}", file=sys.stderr)
    if rows.empty:
        print("ERROR: no clusters selected", file=sys.stderr)
        sys.exit(1)

    sort_col = "total_seqlets" if "total_seqlets" in rows.columns else "cluster_final"
    rows = rows.sort_values(sort_col, ascending=False).reset_index(drop=True)

    cols = [
        c for c in ("cluster_final", "motif", "posneg", "prevalence", "n_groups",
                    "sole_group", "total_seqlets", "n_motifs", "seqlets_per_motif",
                    "jaspar_name", "jaspar_score", "logo")
        if c in rows.columns
    ]
    out = rows[cols].copy()
    out["class"] = ""
    out["notes"] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_annotation_{args.head}"
    tsv = stem.with_name(f"{stem.name}_scaffold.tsv")
    with open(tsv, "w") as f:
        f.write(
            "# Fill the 'class' column, then pass this file to "
            "--annotation-tsv.\n# Suggested classes: "
            + ", ".join(SUGGESTED_CLASSES) + "\n"
        )
        out.to_csv(f, sep="\t", index=False)
    print(f"Saved {tsv}  ({len(out)} rows)", file=sys.stderr)

    if "logo" in out.columns and out["logo"].notna().any():
        write_scaffold_html(rows, stem.with_name(f"{stem.name}_scaffold.html"), args.head)
    else:
        print(
            "  no logo paths resolved, so no HTML written; pass --logo-paths "
            "/ --logo-root",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
