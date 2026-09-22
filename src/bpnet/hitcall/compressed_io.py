"""Compressed-aware path resolution and writers for hit-call tables.

Hit-call outputs are large and numerous -- one set per (experiment, head,
trim configuration) across all 224 experiments, not just the QC-passing
subset the atlas analyses use -- so the tables are stored compressed: gzip
for `.tsv`, bgzip for `.bed` and `.narrowPeak`.

bgzip rather than gzip for the interval formats because bgzip output is
block-compressed and therefore tabix-indexable, while remaining a valid gzip
stream that `gzip`, `pandas` and `polars` all read transparently. Nothing
indexes them today; the point is that compressing them now does not foreclose
it.

**Reads accept either form.** `resolve()` takes the logical path -- the
uncompressed name -- and returns whichever of it and its `.gz` sibling
exists, preferring the compressed one. Every reader goes through it, because
the tree holds a mix: builds predating compression, Fi-NeMo output compressed
after the fact, and freshly written `.gz`. Requiring one form would break the
other.

This matters more than it looks. `cleanup_hitcalls.py` originally compressed
only `hits.bed`, and said why: nothing read it back by exact filename, "so
gzipping it can't break any downstream script's file resolution the way
gzipping hits_unique.tsv would". That hazard is what `resolve()` removes.

Fi-NeMo's own writers are not affected -- it writes `hits.tsv`,
`hits_unique.tsv`, `hits.bed` and `motif_report.tsv` uncompressed, and those
get compressed afterward. `peaks.narrowPeak` is ours but is *input* to
`finemo extract-regions`, which reads it with `polars.scan_csv`; that handles
gzip, so a compressed cache works. Verify on the cluster if polars there is
old, since the fallback is silent breakage at region extraction.
"""

import gzip
import shutil
import subprocess
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

# Written gzip; pandas infers the codec from the suffix on both ends.
GZIP_SUFFIXES = (".tsv", ".txt")
# Written bgzip, for tabix-indexability.
BGZIP_SUFFIXES = (".bed", ".narrowPeak")
# Overridden in tests to simulate a failing compressor.
BGZIP_ARGS = ["-c"]


def compressed_name(path):
    """The `.gz` sibling of a logical path, or the path if already `.gz`."""
    path = Path(path)
    if path.name.endswith(".gz"):
        return path
    return path.with_name(path.name + ".gz")


def plain_name(path):
    """The uncompressed sibling of a path, or the path if already plain."""
    path = Path(path)
    if path.name.endswith(".gz"):
        return path.with_name(path.name[:-3])
    return path


def resolve(path, missing_ok=False):
    """The existing form of `path`, compressed or not, preferring compressed.

    Args:
        path: logical path, with or without a `.gz` suffix.
        missing_ok: return `None` instead of raising when neither exists.

    Raises:
        FileNotFoundError: naming *both* candidates, so the message is
            actionable whichever form the caller expected.

    A warning is issued when both forms exist. That is a real state -- an
    interrupted compression pass, or a rerun writing `.gz` beside an older
    plain file -- and the two can disagree, so silently taking the compressed
    one would hide a stale-data bug.
    """
    compressed, plain = compressed_name(path), plain_name(path)
    if compressed.exists() and plain.exists():
        warnings.warn(
            f"both {compressed} and {plain} exist; using the compressed one. "
            "They may disagree -- delete whichever is stale."
        )
        return compressed
    if compressed.exists():
        return compressed
    if plain.exists():
        return plain
    if missing_ok:
        return None
    raise FileNotFoundError(f"neither {compressed} nor {plain} exists")


def exists(path):
    """Whether either form of `path` is present."""
    return resolve(path, missing_ok=True) is not None


@contextmanager
def ensure_plain(path):
    """Yield a plain-text-readable path for `path`, decompressing to a
    temporary file first if only its `.gz` form exists.

    For handing a small mapping/config file to code that doesn't itself
    decompress gzip -- e.g. Fi-NeMo's own -R/-T mapping-file CLI args and
    `finemo.data_io.load_mapping_tuple`, unlike `polars.scan_csv` (used for
    `peaks.narrowPeak`), which decompresses transparently regardless (see
    module docstring). Only ever used for tiny files (a handful of
    motif_name<TAB>... rows), so decompressing to a temp file on every call
    is cheap. The temp file is removed on exit; the original resolved path
    is yielded as-is (no cleanup) when it was already plain.
    """
    resolved = resolve(path)
    if resolved.suffix != ".gz":
        yield resolved
        return
    tmp = tempfile.NamedTemporaryFile(
        suffix=plain_name(resolved).suffix, delete=False
    )
    try:
        with gzip.open(resolved, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def write_tsv(frame, path, **kwargs):
    """Write a DataFrame as gzipped TSV, returning the path written.

    `path` may be given with or without `.gz`; the output is always
    compressed. Separator and `index=False` are applied unless overridden, so
    call sites do not repeat them.
    """
    out = compressed_name(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("index", False)
    frame.to_csv(out, **kwargs)
    return out


def bgzip_path():
    """The `bgzip` executable, or None.

    Provided by `htslib` in `environment.yml` rather than as a Python
    dependency, matching the project's split: `pyproject.toml` for Python
    packages, the conda environment for command-line tools.
    """
    return shutil.which("bgzip")


def write_bgzip(frame, path, **kwargs):
    """Write a DataFrame as a bgzipped interval file, returning the path.

    Raises rather than falling back to plain gzip when `bgzip` is missing. A
    plain-gzip file named `.bed.gz` is indistinguishable until something tries
    to tabix-index it, and then fails somewhere far from here.

    Interval formats are headerless, so `header=False` is the default.
    """
    out = compressed_name(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("index", False)
    kwargs.setdefault("header", False)

    bgzip = bgzip_path()
    if bgzip is None:
        raise RuntimeError(
            f"bgzip not found on PATH, needed to write {out}. It comes from "
            "htslib in environment.yml: `mamba env update -f environment.yml` "
            "(or `mamba install -c bioconda htslib`). Not falling back to "
            "plain gzip, because the result would be named .gz but not be "
            "tabix-indexable."
        )

    text = frame.to_csv(**kwargs)
    with open(out, "wb") as handle:
        result = subprocess.run(
            [bgzip, *BGZIP_ARGS], input=text.encode(), stdout=handle,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"bgzip failed writing {out}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return out
