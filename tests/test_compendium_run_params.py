"""Tests for the MotifCompendium build parameter record.

`cluster_motifs.py` cannot be imported here -- it imports MotifCompendium at
module level, which is a separate research environment not installed in this
project -- so the parameter logic deliberately lives in `run_params.py`, which
imports nothing beyond the standard library. That is what makes it testable at
all, and it is the reason these tests exist in a file of their own.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "motifcompendium"))

import run_params as rp  # noqa: E402

FIXED = datetime(2026, 9, 11, 12, 0, 0, tzinfo=timezone.utc)


def make(**overrides):
    kwargs = dict(
        within_threshold=0.95,
        across_threshold=0.90,
        min_reads=0,
        blacklist=["ENCSR973QQI"],
        experiments_selected=["ENCSR002", "ENCSR001"],
        experiments_with_modisco=["ENCSR001"],
        out_dir="motifcompendium/bpnet",
        jaspar_path="data/JASPAR2026.txt",
        motifcompendium_version="0.3.1",
        commit="abc123",
        now=FIXED,
    )
    head = overrides.pop("head", "count")
    kwargs.update(overrides)
    return rp.build_run_params(head, **kwargs)


def test_records_both_experiment_sets_sorted():
    p = make()
    # Selected vs actually-contributing is the 219-vs-198 distinction, so both
    # must be present, not just the min_reads that produced them.
    assert p["experiments_selected"] == ["ENCSR001", "ENCSR002"]
    assert p["n_experiments_selected"] == 2
    assert p["experiments_with_modisco"] == ["ENCSR001"]
    assert p["n_experiments_with_modisco"] == 1


def test_accepts_a_keys_view_for_experiments():
    # cluster_motifs.py passes h5_paths.keys(), not a list.
    p = make(experiments_with_modisco={"B": 1, "A": 2}.keys())
    assert p["experiments_with_modisco"] == ["A", "B"]


def test_records_the_thresholds_that_change_the_partition():
    p = make(within_threshold=0.95, across_threshold=0.85)
    assert p["within_threshold"] == 0.95
    assert p["across_threshold"] == 0.85


def test_starts_incomplete():
    p = make()
    assert p["completed"] is False
    assert p["completed_at"] is None
    assert p["started_at"] == FIXED.isoformat()


def test_mark_completed_does_not_mutate_the_original():
    p = make()
    done = rp.mark_completed(p, now=FIXED)
    assert p["completed"] is False          # original untouched
    assert done["completed"] is True
    assert done["completed_at"] == FIXED.isoformat()
    assert done["across_threshold"] == p["across_threshold"]


def test_round_trip_through_disk(tmp_path):
    p = make()
    path = rp.write_run_params(tmp_path / "sub" / "params.json", p)
    assert path.exists()                     # parent created
    assert rp.load_run_params(path) == p


def test_written_json_is_stable_across_writes(tmp_path):
    # sort_keys, so a diff between two builds' files is readable.
    a = rp.write_run_params(tmp_path / "a.json", make()).read_text()
    b = rp.write_run_params(tmp_path / "b.json", make()).read_text()
    assert a == b
    assert json.loads(a)["head"] == "count"


def test_params_path_is_per_head():
    assert rp.params_path(Path("out"), "count").name == (
        "motifcompendium_count_parameters.json"
    )
    assert rp.params_path(Path("out"), "profile").name == (
        "motifcompendium_profile_parameters.json"
    )


def test_compare_finds_the_threshold_difference():
    a = make(across_threshold=0.85)
    b = make(across_threshold=0.90)
    diffs = rp.compare_run_params(a, b)
    assert diffs["across_threshold"] == (0.85, 0.90)


def test_compare_ignores_timestamps_by_default():
    a = make(now=FIXED)
    b = make(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    # Two runs always differ in start time; that must not read as a settings
    # difference, or the signal is buried by noise.
    assert rp.compare_run_params(a, b) == {}


def test_compare_can_be_asked_for_every_key():
    a = make(now=FIXED)
    b = make(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    diffs = rp.compare_run_params(a, b, keys=sorted(set(a) | set(b)))
    assert "started_at" in diffs


def test_compare_reports_missing_keys_rather_than_matching_silently():
    a = make()
    b = {k: v for k, v in make().items() if k != "across_threshold"}
    diffs = rp.compare_run_params(a, b)
    assert diffs["across_threshold"] == (0.90, "<absent>")


def test_compare_detects_a_different_experiment_set_at_equal_thresholds():
    a = make(experiments_with_modisco=["A", "B", "C"])
    b = make(experiments_with_modisco=["A", "B"])
    diffs = rp.compare_run_params(a, b)
    assert "experiments_with_modisco" in diffs
    assert "n_experiments_with_modisco" in diffs


def test_format_comparison_spells_out_which_experiments_differ():
    a = make(experiments_with_modisco=["A", "B", "C"])
    b = make(experiments_with_modisco=["A", "B"])
    text = rp.format_comparison(a, b, "old", "new")
    # "these lists differ" is useless; which ones is the actual question.
    assert "only in old" in text
    assert "C" in text


def test_format_comparison_says_so_when_settings_match():
    text = rp.format_comparison(make(), make(), "a", "b")
    assert "identical" in text


def test_format_record_warns_about_an_unfinished_build():
    text = rp.format_record(make(), "params.json")
    assert "did not finish" in text
    done = rp.format_record(rp.mark_completed(make()), "params.json")
    assert "did not finish" not in done


def test_long_experiment_lists_are_abbreviated():
    p = make(experiments_with_modisco=[f"E{i:03d}" for i in range(200)])
    text = rp.format_record(p, "x")
    assert "200 total" in text or "200 with modisco" in text
    assert len(text.splitlines()) < 30      # not 200 lines of experiment ids


def test_git_commit_never_raises_outside_a_checkout(tmp_path):
    # Provenance capture must not be able to kill a multi-hour build.
    assert rp.git_commit(tmp_path) is None or isinstance(rp.git_commit(tmp_path), str)


def test_git_commit_finds_this_repo():
    sha = rp.git_commit(REPO_ROOT)
    assert sha is None or len(sha) == 40


def test_cli_summarizes_one_record(tmp_path):
    path = rp.write_run_params(tmp_path / "p.json", rp.mark_completed(make()))
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/bpnet/motifcompendium/run_params.py"),
         str(path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "across_threshold: 0.9" in out.stdout


def test_cli_compares_two_records(tmp_path):
    a = rp.write_run_params(tmp_path / "a.json", make(across_threshold=0.85))
    b = rp.write_run_params(tmp_path / "b.json", make(across_threshold=0.90))
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/bpnet/motifcompendium/run_params.py"),
         str(a), str(b)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "across_threshold" in out.stdout


def test_cli_points_at_the_default_change_when_a_record_is_missing(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/bpnet/motifcompendium/run_params.py"),
         str(tmp_path / "nope.json")],
        capture_output=True, text=True,
    )
    assert out.returncode == 1
    # Builds predating this feature have no record; the error has to say what
    # to do instead rather than just "not found".
    assert "2026-08-21" in out.stderr


def test_cli_rejects_three_paths(tmp_path):
    paths = [str(rp.write_run_params(tmp_path / f"{i}.json", make())) for i in range(3)]
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/bpnet/motifcompendium/run_params.py"),
         *paths],
        capture_output=True, text=True,
    )
    assert out.returncode != 0


@pytest.mark.parametrize("key", list(rp.SETTING_KEYS))
def test_every_setting_key_is_actually_recorded(key):
    # A key in SETTING_KEYS that build_run_params never writes would compare
    # as "<absent>" against every other build and look like a difference.
    assert key in make()
