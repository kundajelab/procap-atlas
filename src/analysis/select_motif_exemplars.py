#!/usr/bin/env python3
"""Pick the lineage-restricted and ubiquitous motifs worth showing, and render
their logos side by side for inspection.

The concentration test says the lexicon is tissue-structured; it does not say
which motifs to put in a figure. Reading them off the per-cluster table by eye
is where this goes wrong, because two kinds of cluster look alike in a sorted
list and are not alike at all:

1. A genuine lineage motif -- OCT4::SOX2 in stem/iPSC (31,323 seqlets over 4
   experiments), NEUROG2 in neural (22,570 over 3), RELA in blood (10,101 over
   6). Low prevalence because few experiments are of that lineage, but deeply
   supported within them.

2. A low-abundance split of a ubiquitous motif. Cluster 192 is labelled NFYA,
   sits in blood_immune alone, and would read as "blood-specific NFYA" -- but
   cluster 1 is also NFYA, spans all 19 tissue groups and carries 1.5M seqlets
   against cluster 192's 154. It is a fragment of the ubiquitous CCAAT motif,
   not a lineage-restricted one. Same for SP9 (110 seqlets vs cluster 0's
   3.87M), Atf1, Nrf1, ELF2, ZNF131 and TFEC.

So a restricted cluster whose JASPAR name *also* labels a broad cluster is
reported but flagged, and excluded from the selection by default. On the real
count head that is 8 of the 45 single-group TF-matched clusters. Without the
check, three of the eight would have been picked as per-tissue exemplars
(heart and metastatic_carcinoma have no better single-group candidate than a
~150-seqlet Atf1 fragment), and a reviewer comparing panel logos against the
ubiquitous ones would have spotted it immediately.

The seqlet floor does related work: it is what separates SPIB (1,689 seqlets
over 11 blood experiments) from SPI1 (44 seqlets over 2), which carry equally
suggestive names.

Usage:
    python src/analysis/select_motif_exemplars.py --head count
    python src/analysis/select_motif_exemplars.py --head count --min-seqlets 5000
    python src/analysis/select_motif_exemplars.py --head count --per-group 2 --keep-shared-name
"""

import argparse
import base64
import html
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
DEFAULT_OUT = REPO_ROOT / "figures" / "motif_atlas"

# A cluster in this many or more tissue groups counts as broad, for selecting
# the ubiquitous row. Set below the 19-group maximum so a motif missing from a
# couple of groups still counts as broad.
BROAD_GROUP_FLOOR = 15

# A *lower* floor for the separate question of whether a restricted cluster's
# JASPAR name is one the atlas discovers broadly elsewhere. These two are not
# the same threshold, and conflating them let CTCF through: cluster 87 sits in
# 3 tissue groups with 63,421 seqlets and reads as a lineage motif, while
# cluster 15 carries the same name across 13 groups. 13 clears 10 but not 15,
# so the single floor flagged nothing and CTCF -- the textbook ubiquitous
# architectural factor -- became a lineage-restricted candidate.
#
# A seqlet-ratio rule would not have caught it either: cluster 87 has *more*
# seqlets than the 13-group cluster 15 (63,421 vs 38,508), so this is not a
# low-abundance split. The discriminating fact is simply that the atlas
# discovers CTCF broadly somewhere, which makes any narrow CTCF cluster a poor
# lineage claim regardless of its support.
SPLIT_FLAG_GROUP_FLOOR = 10


def load_concentration(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    required = {"cluster_final", "jaspar_name", "prevalence", "n_groups",
                "total_seqlets", "motif_class"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{path} lacks {sorted(missing)}; it should be "
            "motif_concentration_{head}_tissue.tsv from "
            "motif_group_concentration.py"
        )
    return d


def broad_names(d: pd.DataFrame, floor: int = BROAD_GROUP_FLOOR) -> set:
    """JASPAR names carried by at least one cluster spanning `floor`+ groups."""
    return set(d[d["n_groups"] >= floor]["jaspar_name"].dropna())


def annotate(
    d: pd.DataFrame,
    floor: int = BROAD_GROUP_FLOOR,
    split_floor: int = SPLIT_FLAG_GROUP_FLOOR,
) -> pd.DataFrame:
    out = d.copy()
    out["name_also_broad"] = out["jaspar_name"].isin(broad_names(out, split_floor))
    out["lineage"] = lineage_label(out)
    return out


def lineage_label(d: pd.DataFrame) -> pd.Series:
    """The tissue grouping a cluster is confined to, single-group or not.

    `sole_group` is NaN whenever a cluster spans more than one group, and
    pandas `groupby` drops NaN keys -- so grouping the per-group cap on
    `sole_group` silently discarded every multi-group candidate, making
    --max-groups greater than 1 do nothing at all.

    That mattered more than a stray flag. A single-group criterion cannot
    detect a factor whose lineage spans several of the 19 keyword groups, and
    the clearest lineage motifs in the atlas are exactly those: MEF2A sits in
    {heart, muscle} (17 experiments, concentration 0.211 -- the third most
    concentrated cluster in the lexicon), HNF1B in {gi_tract, liver_biliary,
    pancreas, metastatic_carcinoma}, i.e. endoderm. Both read as "not
    restricted" under n_groups == 1 while being textbook lineage factors.
    """
    sole = d["sole_group"] if "sole_group" in d.columns else pd.Series(
        [None] * len(d), index=d.index
    )
    groups = d["groups"] if "groups" in d.columns else pd.Series(
        [""] * len(d), index=d.index
    )
    return sole.where(sole.notna() & (sole.astype(str) != ""), groups).astype(str)


def select_restricted(
    d: pd.DataFrame,
    max_groups: int = 1,
    min_prevalence: int = 2,
    min_seqlets: float = 1000,
    per_group: int = 3,
    keep_shared_name: bool = False,
    tf_only: bool = True,
    sort_by: str = "prevalence",
) -> pd.DataFrame:
    """Lineage candidates: few groups, enough support, name not also broad.

    Ranking matters more than it looks. By seqlets, blood_immune's top three
    are POU2F3/RELA/GATA2 and GATA2 sits at prevalence 2 -- deeply supported
    inside two experiments. By prevalence, SPIB comes up instead at 11 of the
    41 blood experiments. For a figure claiming lineage specificity the second
    is the better evidence: recurrence across many experiments of one lineage
    is the claim, while depth within two is consistent with a single peculiar
    sample. Hence prevalence is the default, seqlets available for the cases
    where absolute support is the question.
    """
    sub = d
    if tf_only:
        sub = sub[sub["motif_class"] == "TF-matched"]
    sub = sub[
        (sub["n_groups"] <= max_groups)
        & (sub["prevalence"] >= min_prevalence)
        & (sub["total_seqlets"] >= min_seqlets)
    ]
    if not keep_shared_name:
        sub = sub[~sub["name_also_broad"]]
    order = ["prevalence", "total_seqlets"] if sort_by == "prevalence" else [
        "total_seqlets", "prevalence"
    ]
    sub = sub.sort_values(order, ascending=False)
    if per_group > 0 and "lineage" in sub.columns:
        sub = sub.groupby("lineage", group_keys=False).head(per_group)
    return sub.sort_values(["lineage", "total_seqlets"], ascending=[True, False])


def select_ubiquitous(
    d: pd.DataFrame,
    min_groups: int = BROAD_GROUP_FLOOR,
    top_n: int = 10,
    tf_only: bool = True,
    one_per_name: bool = True,
) -> pd.DataFrame:
    sub = d
    if tf_only:
        sub = sub[sub["motif_class"] == "TF-matched"]
    sub = sub[sub["n_groups"] >= min_groups]
    sub = sub.sort_values("total_seqlets", ascending=False)
    if one_per_name:
        # The lexicon carries near-duplicate broad clusters -- ELF2 and Atf1
        # each appear twice among the top ten by seqlets -- so an unfiltered
        # list spends figure rows on the same motif twice.
        sub = sub.drop_duplicates(subset="jaspar_name", keep="first")
    return sub.head(top_n)


def embed_svg(path: Path) -> str | None:
    """An SVG as a base64 data URI, or None if unreadable.

    Embedded rather than linked: this HTML gets copied off the machine that
    made it, and a linked logo breaks the moment it does.
    """
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(
            Path(path).read_bytes()
        ).decode("ascii")
    except OSError:
        return None


def resolve_logos(d: pd.DataFrame, logo_paths: Path | None, logo_root: Path) -> dict:
    """cluster_final -> data URI, for whichever logos can be found."""
    if logo_paths is None or not Path(logo_paths).exists():
        return {}
    lp = pd.read_csv(logo_paths, sep="\t")
    col = "logo_fwd_svg" if "logo_fwd_svg" in lp.columns else None
    if col is None:
        return {}
    out = {}
    for row in lp.itertuples():
        rel = getattr(row, col, None)
        if not isinstance(rel, str):
            continue
        uri = embed_svg(Path(logo_root) / rel)
        if uri:
            out[row.cluster_final] = uri
    return out


def _row_html(r, logos: dict) -> str:
    uri = logos.get(r.cluster_final)
    img = (
        f'<img src="{uri}" style="height:58px">' if uri
        else '<span style="color:#999">logo unavailable</span>'
    )
    name = html.escape(str(r.jaspar_name))
    group = html.escape(str(getattr(r, "lineage", "") or ""))
    flag = (
        ' <span style="color:#b2182b;font-weight:bold">NAME ALSO BROAD</span>'
        if getattr(r, "name_also_broad", False) else ""
    )
    return f"""<tr>
  <td>{img}</td>
  <td><b>{name}</b>{flag}<br><span style="color:#666">cluster {r.cluster_final}</span></td>
  <td>{group}</td>
  <td style="text-align:right">{int(r.prevalence)}</td>
  <td style="text-align:right">{int(r.n_groups)}</td>
  <td style="text-align:right">{int(r.total_seqlets):,}</td>
</tr>"""


def write_html(restricted, ubiquitous, flagged, logos, path: Path, head: str) -> Path:
    def table(df, caption):
        if not len(df):
            return f"<h2>{caption}</h2><p>none</p>"
        rows = "\n".join(_row_html(r, logos) for r in df.itertuples())
        return f"""<h2>{caption}</h2>
<table cellpadding="6" style="border-collapse:collapse">
<tr style="background:#eee"><th>logo</th><th>JASPAR</th><th>tissue group</th>
<th>experiments</th><th>groups</th><th>seqlets</th></tr>
{rows}
</table>"""

    banner = ""
    if not logos:
        banner = (
            '<p style="background:#fff3cd;padding:10px">No logos could be read. '
            "Pass --logo-paths and --logo-root pointing at the compendium's "
            "cluster_logo_paths.tsv and the directory its paths are relative "
            "to.</p>"
        )

    body = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{head} motif exemplars</title></head>
<body style="font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto">
<h1>{head} head: motif exemplars</h1>
{banner}
<p>Restricted candidates are TF-matched clusters confined to one tissue group
with enough seqlet support to be more than noise, whose JASPAR name is not also
carried by a broad cluster. See the flagged table for why that last condition
matters.</p>
{table(restricted, "Lineage-restricted candidates")}
{table(ubiquitous, "Ubiquitous motifs")}
{table(flagged, "Excluded: restricted, but the name also labels a broad cluster")}
</body></html>"""
    path.write_text(body)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", default="count", choices=["profile", "count"])
    parser.add_argument(
        "--concentration-tsv", type=Path, default=None, metavar="PATH",
        help="motif_concentration_{head}_tissue.tsv (default: under --out-dir)",
    )
    parser.add_argument(
        "--logo-paths", type=Path, default=None, metavar="PATH",
        help="motifcompendium_{head}_cluster_logo_paths.tsv",
    )
    parser.add_argument(
        "--logo-root", type=Path, default=None, metavar="DIR",
        help="directory the logo paths are relative to (default: their own parent)",
    )
    parser.add_argument(
        "--max-groups", type=int, default=1, metavar="N",
        help="a candidate may span at most this many tissue groups (default: 1)",
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=2, metavar="N",
        help="minimum experiments (default: 2; prevalence-1 clusters are "
             "single-group by construction and carry no information)",
    )
    parser.add_argument(
        "--min-seqlets", type=float, default=1000, metavar="N",
        help="minimum total seqlets (default: 1000). This is what separates "
             "SPIB at 1,689 over 11 experiments from SPI1 at 44 over 2",
    )
    parser.add_argument(
        "--per-group", type=int, default=3, metavar="N",
        help="keep at most this many per lineage (the tissue group, or the "
             "set of groups when --max-groups > 1), best-supported first; "
             "0 keeps all (default: 3)",
    )
    parser.add_argument(
        "--broad-groups", type=int, default=BROAD_GROUP_FLOOR, metavar="N",
        help=f"groups a cluster must span to be shown as ubiquitous "
             f"(default: {BROAD_GROUP_FLOOR})",
    )
    parser.add_argument(
        "--split-flag-groups", type=int, default=SPLIT_FLAG_GROUP_FLOOR,
        metavar="N",
        help="a restricted cluster is flagged if another cluster with the same "
             "JASPAR name spans at least this many groups, i.e. the atlas "
             f"discovers that factor broadly elsewhere (default: "
             f"{SPLIT_FLAG_GROUP_FLOOR})",
    )
    parser.add_argument(
        "--top-ubiquitous", type=int, default=10, metavar="N",
        help="how many ubiquitous motifs to show (default: 10)",
    )
    parser.add_argument(
        "--keep-shared-name", action="store_true",
        help="do not exclude restricted clusters whose name also labels a "
             "broad cluster. Off by default: on real data these are splits of "
             "a ubiquitous motif, not lineage motifs",
    )
    parser.add_argument(
        "--sort-by", default="prevalence", choices=["prevalence", "seqlets"],
        help="rank restricted candidates by experiments of the lineage they "
             "recur in (default) or by absolute seqlet support",
    )
    parser.add_argument(
        "--keep-duplicate-names", action="store_true",
        help="do not collapse ubiquitous clusters sharing a JASPAR name; by "
             "default only the best-supported is shown, since near-duplicate "
             "broad clusters otherwise fill the list",
    )
    parser.add_argument("--include-unmatched", action="store_true",
                        help="also consider clusters JASPAR could not name")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, metavar="DIR")
    args = parser.parse_args()

    conc = args.concentration_tsv or (
        args.out_dir / f"motif_concentration_{args.head}_tissue.tsv"
    )
    if not conc.exists():
        print(f"ERROR: {conc} not found", file=sys.stderr)
        print("Run motif_group_concentration.py --group-level tissue first.",
              file=sys.stderr)
        sys.exit(1)

    d = annotate(load_concentration(conc), args.broad_groups,
                 args.split_flag_groups)
    tf_only = not args.include_unmatched

    restricted = select_restricted(
        d, args.max_groups, args.min_prevalence, args.min_seqlets,
        args.per_group, args.keep_shared_name, tf_only, args.sort_by,
    )
    ubiquitous = select_ubiquitous(
        d, args.broad_groups, args.top_ubiquitous, tf_only,
        one_per_name=not args.keep_duplicate_names,
    )

    flagged = d[
        (d["n_groups"] <= args.max_groups)
        & (d["prevalence"] >= args.min_prevalence)
        & d["name_also_broad"]
    ].sort_values("total_seqlets", ascending=False)

    logo_paths = args.logo_paths
    logo_root = args.logo_root or (
        Path(logo_paths).parent if logo_paths else MC_DIR
    )
    logos = resolve_logos(d, logo_paths, logo_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_exemplars_{args.head}"
    restricted.to_csv(f"{stem}_restricted.tsv", sep="\t", index=False)
    ubiquitous.to_csv(f"{stem}_ubiquitous.tsv", sep="\t", index=False)
    html_path = write_html(
        restricted, ubiquitous, flagged, logos, Path(f"{stem}.html"), args.head
    )

    cols = ["cluster_final", "jaspar_name", "lineage", "prevalence",
            "n_groups", "total_seqlets"]
    print(f"Lineage-restricted candidates ({len(restricted)}):")
    print(restricted[cols].to_string(index=False))
    print(f"\nUbiquitous ({len(ubiquitous)}):")
    print(ubiquitous[[c for c in cols if c != "lineage"]].to_string(index=False))
    if len(flagged):
        print(
            f"\nExcluded {len(flagged)} restricted cluster(s) whose JASPAR name "
            "also labels a broad cluster (splits of a ubiquitous motif):"
        )
        print(flagged[cols].to_string(index=False))
    if not logos:
        print("\nWARNING: no logos embedded; pass --logo-paths/--logo-root",
              file=sys.stderr)
    print(f"\nSaved {stem}_restricted.tsv, {stem}_ubiquitous.tsv, {html_path}")


if __name__ == "__main__":
    main()
