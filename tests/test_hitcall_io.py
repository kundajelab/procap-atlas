"""Tests for src/bpnet/hitcall/compressed_io.py.

The read path is what carries risk. Hit-call trees hold a mix of formats --
builds predating compression, Fi-NeMo output compressed after the fact, and
freshly written .gz -- so resolution has to accept either form and must not
silently prefer a stale file.
"""

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src" / "bpnet" / "hitcall")
)

import compressed_io as hio  # noqa: E402

FRAME = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 20],
                      "end": [15, 25]})


# --- name helpers ------------------------------------------------------------


@pytest.mark.parametrize("given,expected", [
    ("hits.tsv", "hits.tsv.gz"),
    ("hits.tsv.gz", "hits.tsv.gz"),      # idempotent
    ("peaks.narrowPeak", "peaks.narrowPeak.gz"),
])
def test_compressed_name(given, expected):
    assert hio.compressed_name(Path(given)).name == expected


@pytest.mark.parametrize("given,expected", [
    ("hits.tsv.gz", "hits.tsv"),
    ("hits.tsv", "hits.tsv"),            # idempotent
])
def test_plain_name(given, expected):
    assert hio.plain_name(Path(given)).name == expected


# --- resolve -----------------------------------------------------------------


def test_resolve_finds_the_compressed_form_from_the_logical_path(tmp_path):
    """Callers pass the uncompressed name they have always used."""
    target = tmp_path / "hits.tsv.gz"
    target.write_bytes(b"")
    assert hio.resolve(tmp_path / "hits.tsv") == target


def test_resolve_finds_the_plain_form(tmp_path):
    """Builds predating compression must keep working."""
    target = tmp_path / "hits.tsv"
    target.write_text("x")
    assert hio.resolve(tmp_path / "hits.tsv") == target


def test_resolve_accepts_a_gz_path_for_a_plain_file(tmp_path):
    """Resolution is symmetric: a caller holding the .gz name still finds it."""
    target = tmp_path / "hits.tsv"
    target.write_text("x")
    assert hio.resolve(tmp_path / "hits.tsv.gz") == target


def test_resolve_prefers_compressed_and_warns_when_both_exist(tmp_path):
    """A real state after an interrupted compression pass. The two can
    disagree, so this must not be silent."""
    (tmp_path / "hits.tsv").write_text("stale")
    (tmp_path / "hits.tsv.gz").write_bytes(b"")
    with pytest.warns(UserWarning, match="both .* exist"):
        got = hio.resolve(tmp_path / "hits.tsv")
    assert got.name == "hits.tsv.gz"


def test_resolve_error_names_both_candidates(tmp_path):
    """Whichever form the caller expected, the message has to be actionable."""
    with pytest.raises(FileNotFoundError) as excinfo:
        hio.resolve(tmp_path / "hits.tsv")
    message = str(excinfo.value)
    assert "hits.tsv.gz" in message and "hits.tsv" in message


def test_resolve_missing_ok_returns_none(tmp_path):
    assert hio.resolve(tmp_path / "hits.tsv", missing_ok=True) is None


def test_exists_accepts_either_form(tmp_path):
    assert not hio.exists(tmp_path / "hits.tsv")
    (tmp_path / "hits.tsv.gz").write_bytes(b"")
    assert hio.exists(tmp_path / "hits.tsv")


# --- write_tsv ---------------------------------------------------------------


def test_write_tsv_compresses_and_round_trips(tmp_path):
    out = hio.write_tsv(FRAME, tmp_path / "hits.tsv")
    assert out.name == "hits.tsv.gz"
    with gzip.open(out, "rt") as handle:
        assert handle.readline().startswith("chrom\t")
    pd.testing.assert_frame_equal(pd.read_csv(out, sep="\t"), FRAME)


def test_write_tsv_is_idempotent_on_a_gz_path(tmp_path):
    """Must not produce hits.tsv.gz.gz."""
    out = hio.write_tsv(FRAME, tmp_path / "hits.tsv.gz")
    assert out.name == "hits.tsv.gz"


def test_write_tsv_defaults_to_tab_separated_without_an_index(tmp_path):
    out = hio.write_tsv(FRAME, tmp_path / "hits.tsv")
    with gzip.open(out, "rt") as handle:
        header = handle.readline().strip()
    assert header == "chrom\tstart\tend"


def test_write_tsv_defaults_are_overridable(tmp_path):
    out = hio.write_tsv(FRAME, tmp_path / "hits.tsv", header=False)
    with gzip.open(out, "rt") as handle:
        assert handle.readline().startswith("chr1\t")


def test_write_tsv_creates_the_parent_directory(tmp_path):
    out = hio.write_tsv(FRAME, tmp_path / "nested" / "deep" / "hits.tsv")
    assert out.exists()


# --- write_bgzip -------------------------------------------------------------

requires_bgzip = pytest.mark.skipif(
    shutil.which("bgzip") is None, reason="bgzip (htslib) not installed"
)


def test_write_bgzip_raises_a_actionable_error_without_bgzip(tmp_path, monkeypatch):
    """Must not fall back to plain gzip: the file would be named .gz but fail
    to tabix-index, somewhere far from here."""
    monkeypatch.setattr(hio, "bgzip_path", lambda: None)
    with pytest.raises(RuntimeError, match="htslib in environment.yml"):
        hio.write_bgzip(FRAME, tmp_path / "hits.bed")
    assert not (tmp_path / "hits.bed.gz").exists()


def test_write_bgzip_cleans_up_after_a_failure(tmp_path, monkeypatch):
    """A truncated .bed.gz left behind would be picked up by resolve()."""
    # `python -c` with no argument exits 2 -- a portable guaranteed failure.
    monkeypatch.setattr(hio, "bgzip_path", lambda: sys.executable)
    monkeypatch.setattr(hio, "BGZIP_ARGS", ["-c"])
    with pytest.raises(RuntimeError, match="bgzip failed"):
        hio.write_bgzip(FRAME, tmp_path / "hits.bed")
    assert not (tmp_path / "hits.bed.gz").exists()


@requires_bgzip
def test_write_bgzip_writes_headerless_and_round_trips(tmp_path):
    out = hio.write_bgzip(FRAME, tmp_path / "hits.bed")
    assert out.name == "hits.bed.gz"
    got = pd.read_csv(out, sep="\t", header=None)
    assert got.shape == FRAME.shape
    assert list(got.iloc[0]) == ["chr1", 10, 15]


@requires_bgzip
def test_bgzip_output_is_readable_as_plain_gzip(tmp_path):
    """bgzip is a valid gzip stream; every reader downstream relies on this."""
    out = hio.write_bgzip(FRAME, tmp_path / "hits.bed")
    with gzip.open(out, "rt") as handle:
        assert handle.readline().strip() == "chr1\t10\t15"


def is_bgzf(path):
    """BGZF magic: gzip magic, the FEXTRA flag, and htslib's "BC" subfield.

    `bgzip -t` is not a usable check -- it accepts a plain gzip file, so it
    validates decompressibility rather than block structure.
    """
    with open(path, "rb") as handle:
        head = handle.read(16)
    return head[:2] == b"\x1f\x8b" and bool(head[3] & 0x04) and b"BC" in head


@requires_bgzip
def test_bgzip_output_is_block_compressed(tmp_path):
    """The whole reason for bgzip over gzip: tabix-indexability."""
    out = hio.write_bgzip(FRAME, tmp_path / "hits.bed")
    assert is_bgzf(out)

    plain = tmp_path / "plain.bed.gz"
    with gzip.open(plain, "wt") as handle:
        handle.write("chr1\t10\t15\n")
    assert not is_bgzf(plain), "the check must distinguish bgzip from gzip"
