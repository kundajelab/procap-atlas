"""Shared biosample -> tissue/lineage group mapping for atlas-level analyses.

The atlas spans 126 distinct ENCODE biosamples, most represented by a single
experiment (see configs/experiment_config.yaml). Per-biosample columns are
therefore too sparse to read structure off of, and per-biosample means are
dominated by whichever biosamples happen to have many experiments (HCT116
n=16, Metastatic Breast Carcinoma in the Brain n=10, PBMC n=8). Grouping
biosamples into coarser tissue/lineage classes is what makes cross-experiment
motif comparisons interpretable.

Group assignment is inherently curation, not computation. `GROUP_RULES` below
is a keyword-matching first pass, applied in order (first match wins, so more
specific rules must come first). Callers should write the resolved mapping out
with `write_group_tsv` and hand-correct it, then pass it back via
`--biosample-groups`; `load_group_map` prefers an override file over the rules
whenever one is supplied.

Metastases are assigned to their tissue of **origin**, not to a
`metastatic_carcinoma` bucket and not to the organ they spread to. All eight
metastatic biosamples in the atlas name their origin -- "Metastatic Breast
Carcinoma in the Brain", "Colon Carcinoma Metastatic in the Lung" -- and origin
is what determines the regulatory program, so a breast metastasis belongs with
breast.

The earlier single bucket was actively misleading. 21 of 198 experiments landed
in one pseudo-lineage that is a clinical category rather than a tissue, and it
fragmented real ones: HNF1B's groups read `{gi_tract, liver_biliary, pancreas,
metastatic_carcinoma}` -- endoderm plus endoderm-derived metastases, one
lineage counted as four groups -- which inflated `n_groups` for exactly the
factors the analysis exists to detect. It was also uninterpretable in
aggregate, since a bulk metastasis carries tumour, stroma and immune
infiltrate together, so an immune motif appearing "in metastatic carcinoma"
could be infiltrate rather than tumour-intrinsic regulation.

Matching runs against the extracted origin substring alone, which is
load-bearing: the destination organ would otherwise claim the sample, and for
"Metastatic Breast Carcinoma in the Brain" the neural rule precedes the breast
rule, so it would be filed as neural.

Unmatched biosamples land in "other" and are always reported by
`load_group_map`, so a biosample added to the atlas later can never be
silently swept into a group it doesn't belong to.
"""

import re
import sys
from pathlib import Path

# Patterns that pull the tissue of origin out of a metastasis's name. The atlas
# uses two orderings: "Metastatic {origin} Carcinoma in the {destination}" and
# "{origin} Carcinoma Metastatic in the {destination}".
METASTASIS_ORIGIN_PATTERNS: tuple[str, ...] = (
    r"metastatic\s+(.+?)\s+carcinoma",
    r"^(.+?)\s+carcinoma\s+metastatic",
)

# (group, [regex patterns]) applied in order; first match wins.
GROUP_RULES: list[tuple[str, list[str]]] = [
    (
        "blood_immune",
        [
            r"\bt-?cell\b", r"\bb cell\b", r"cytotoxic t", r"helper t",
            r"natural killer", r"monocyte", r"neutrophil", r"macrophage",
            r"peripheral blood", r"\bspleen\b", r"lymph node", r"peyer",
            r"\bk562\b", r"nalm6", r"\breh\b", r"\bsem\b", r"sp-49",
            r"sudhl", r"oci-ly", r"karpas", r"pfeiffer", r"thymus",
        ],
    ),
    (
        "neural",
        [
            r"\bneuron\b", r"astrocyte", r"cerebell", r"cerebral", r"cortex",
            r"spinal cord", r"\bnerve\b", r"luhmes", r"rencell",
            r"neural crest", r"\bbrain\b", r"glia",
        ],
    ),
    (
        "heart",
        [
            r"\bheart\b", r"cardiac", r"myocardium", r"ventricle", r"atrium",
            r"epicardium",
        ],
    ),
    (
        "vascular",
        [
            r"\baorta\b", r"artery", r"\bvein\b", r"vena cava", r"endothelial",
            r"carotid",
        ],
    ),
    (
        "muscle",
        [
            r"skeletal muscle", r"psoas", r"gastrocnemius",
            r"sternocleidomastoid", r"muscularis",
        ],
    ),
    ("liver_biliary", [r"\bliver\b", r"hepat", r"bile duct", r"gallbladder"]),
    (
        "gi_tract",
        [
            r"\bcolon\b", r"colonic", r"\brectum\b", r"duodenum", r"ileum",
            r"jejunum", r"\bstomach\b", r"esophagus", r"gastroesophageal",
            r"appendix", r"caco-2", r"hct116", r"\bintestin",
        ],
    ),
    ("pancreas", [r"pancrea"]),
    (
        "lung_airway",
        [r"\blung\b", r"bronch", r"trachea", r"a549", r"calu3", r"pulmonary"],
    ),
    ("kidney_urinary", [r"\bkidney\b", r"\bureter\b", r"urinary bladder", r"renal"]),
    (
        "reproductive",
        [
            r"\bovary\b", r"\buterus\b", r"\btestis\b", r"prostate", r"\bpenis\b",
            r"\bvagina\b", r"fallopian", r"placenta", r"endometri",
        ],
    ),
    ("breast", [r"\bbreast\b", r"mcf ?10a", r"mammary"]),
    ("endocrine", [r"adrenal", r"thyroid", r"parotid", r"pituitary"]),
    ("adipose", [r"adipose", r"fat pad"]),
    ("skin", [r"\bskin\b", r"keratinocyte", r"epiderm"]),
    ("bone", [r"osteocyte", r"osteoblast", r"a673", r"\bbone\b"]),
    (
        "stem_ipsc",
        [r"wtc11", r"\bh9\b", r"gm23338", r"endodermal", r"\bipsc\b", r"embryonic stem"],
    ),
    ("hek", [r"hek ?293"]),
]

OTHER_GROUP = "other"


def metastasis_origin(biosample: str) -> str | None:
    """The tissue of origin named in a metastasis's biosample name, or None.

    Returns None for non-metastatic biosamples and for metastatic ones whose
    name follows neither ordering, so an unparseable metastasis falls through
    to `other` and gets reported rather than being filed by destination.
    """
    name = (biosample or "").lower()
    if "metasta" not in name:
        return None
    for pattern in METASTASIS_ORIGIN_PATTERNS:
        match = re.search(pattern, name)
        if match:
            origin = match.group(1).strip()
            if origin:
                return origin
    return None


def assign_group(biosample: str) -> str:
    """Assign one biosample name to a tissue/lineage group via GROUP_RULES.

    Metastases are matched on their origin substring alone. Matching the whole
    name would let the destination organ win: the neural rule precedes the
    breast rule, so "Metastatic Breast Carcinoma in the Brain" would be filed
    as neural.
    """
    if not biosample:
        return OTHER_GROUP
    origin = metastasis_origin(biosample)
    haystack = origin if origin is not None else biosample.lower()
    for group, patterns in GROUP_RULES:
        if any(re.search(p, haystack) for p in patterns):
            return group
    return OTHER_GROUP


def load_group_map(
    experiments: dict[str, dict],
    override_tsv: Path | None = None,
    quiet: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve experiment -> group and experiment -> biosample maps.

    `experiments` is configs/experiment_config.yaml's "experiments" mapping.
    `override_tsv` is a curated biosample<TAB>group table (as written by
    `write_group_tsv`); biosamples it doesn't list still fall back to
    GROUP_RULES rather than being dropped.

    Returns (group_map, biosample_map), both keyed by experiment accession.
    Biosamples that fell through to "other" are reported to stderr unless
    `quiet`, since a silent "other" is how a newly added tissue gets
    misgrouped without anyone noticing.
    """
    overrides: dict[str, str] = {}
    if override_tsv is not None:
        if not Path(override_tsv).exists():
            raise FileNotFoundError(f"biosample group table not found: {override_tsv}")
        with open(override_tsv) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] != "biosample":
                    overrides[parts[0]] = parts[1]

    biosample_map, group_map = {}, {}
    for exp_id, exp in experiments.items():
        biosample = exp.get("biosample") or "unknown"
        biosample_map[exp_id] = biosample
        group_map[exp_id] = overrides.get(biosample) or assign_group(biosample)

    if not quiet:
        ungrouped = sorted(
            {biosample_map[e] for e, g in group_map.items() if g == OTHER_GROUP}
        )
        if ungrouped:
            print(
                f"NOTE: {len(ungrouped)} biosample(s) fell through to "
                f"{OTHER_GROUP!r}: {', '.join(ungrouped)}",
                file=sys.stderr,
            )
            print(
                "      Curate them with --write-group-tsv / --biosample-groups.",
                file=sys.stderr,
            )
    return group_map, biosample_map


def write_group_tsv(
    biosample_map: dict[str, str], group_map: dict[str, str], out_path: Path
) -> None:
    """Write an editable biosample<TAB>group<TAB>n_experiments template."""
    counts: dict[tuple[str, str], int] = {}
    for exp_id, biosample in biosample_map.items():
        key = (biosample, group_map[exp_id])
        counts[key] = counts.get(key, 0) + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Edit the group column, then pass this file to --biosample-groups.\n")
        f.write("biosample\tgroup\tn_experiments\n")
        for (biosample, group), n in sorted(counts.items()):
            f.write(f"{biosample}\t{group}\t{n}\n")
    print(f"Wrote editable biosample group table to {out_path}", file=sys.stderr)
