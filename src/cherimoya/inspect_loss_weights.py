#!/usr/bin/env python3
"""Inspect Cherimoya's learned Kendall uncertainty loss weights (lw0/lw1)
across archived training-hyperparameter configs.

Cherimoya.fit() learns per-fold weights balancing profile loss (lw0) against
count loss (lw1) during training: w0 = 1/(2*lw0**2), w1 = 1/(2*lw1**2). Both
start at 1.0 and are optimized by their own SGD optimizer until
torch.abs(self.lw0.grad).mean() < 1, at which point they're frozen for the
rest of training (see cherimoya.py's fit(), pinned commit
8e4283fe56db4a29418c1d8119da3240d7c709ba -- see cherimoya.def/setup_env.sh
for why an exact commit rather than a tag or PyPI version).

This was written to test the upstream author's hypothesis for why longer
training (uncapped max_epochs, no early_stopping) shifts benchmark metrics
toward profile shape at some cost to count correlation: profile loss has
larger raw magnitude, so the automatic weighting could keep "slowly
squeezing" w0 up relative to w1 before freezing, given enough epochs. That
hypothesis did NOT hold up empirically: w0/w1 came back nearly identical
(~0.025-0.028) across every config already benchmarked (20_5_2, 100_None_5,
current), with best and final checkpoints matching almost exactly within
each config -- these weights freeze very early and don't meaningfully drift
regardless of run length. The loss weights are also not what selects the
"best" checkpoint at all: fit() saves *.torch whenever
valid_count_corr > best_corr (a bare validation-set count-correlation
comparison, evaluated on EMA-averaged weights), not by loss. The
count-metric regression is more likely explained by that selection rule
itself getting more chances to overfit the validation set with more
epochs/no early stopping, not by the loss weighting.

This script loads saved checkpoints directly (CPU-only, no GPU needed) and
reports lw0, lw1, the implied w0/w1 ratio, and whether each is frozen
(requires_grad == False), for both the "best" (*.torch) and "final"
(*.final.torch, EMA weights) checkpoint per fold -- the two can differ if
freezing happens after the best-count-corr epoch, which matters because
benchmark_cherimoya.py loads *.torch (best), not *.final.torch.

Must run in a Cherimoya environment (needs triton importable, even though
loading itself is CPU-only) -- see src/cherimoya/README.md's Prerequisites.

Usage:
    python src/cherimoya/inspect_loss_weights.py
    python src/cherimoya/inspect_loss_weights.py --configs current _20_5_2
    python src/cherimoya/inspect_loss_weights.py --model-dir-root /scratch/users/ayhe/procap-atlas/models/cherimoya
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
DEFAULT_MODEL_ROOT = REPO_ROOT / "models" / "cherimoya"

# Oldest to newest; "" is the current (un-prefixed) model directory.
DEFAULT_CONFIGS = ["_20_5_2", "_100_None_5", ""]


def load_weights(path):
    """Return (lw0, lw1, lw0_frozen) from a saved checkpoint, or None if
    missing. CPU-only -- map_location avoids needing a GPU just to read two
    scalars.

    fit_cherimoya.py's Cherimoya.fit() pickles the whole model via
    torch.save(self, ...), so the common case is an object with .lw0/.lw1
    attributes. Some checkpoints instead turn out to be a plain state_dict
    (torch.save(self.state_dict(), ...)), in which case lw0/lw1 show up as
    top-level tensor entries instead, and "frozen" isn't recoverable (a
    state_dict tensor carries no requires_grad history).
    """
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if hasattr(obj, "lw0"):
        return obj.lw0.item(), obj.lw1.item(), not obj.lw0.requires_grad

    if isinstance(obj, dict):
        candidates = [obj]
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(obj.get(key), dict):
                candidates.append(obj[key])
        for candidate in candidates:
            if "lw0" in candidate and "lw1" in candidate:
                lw0, lw1 = candidate["lw0"], candidate["lw1"]
                lw0 = lw0.item() if hasattr(lw0, "item") else float(lw0)
                lw1 = lw1.item() if hasattr(lw1, "item") else float(lw1)
                return lw0, lw1, None
        raise TypeError(
            f"{path}: loaded a dict but couldn't find lw0/lw1 under top-level "
            f"keys {sorted(obj.keys())} (or nested under state_dict/model/"
            "model_state_dict) -- inspect this checkpoint's structure and "
            "update load_weights() to match"
        )

    raise TypeError(f"{path}: loaded unexpected type {type(obj)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"parent of each config's model directory (default: {DEFAULT_MODEL_ROOT})",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help=(
            "config subdirectory names under --model-dir-root, oldest to "
            f"newest; '' means --model-dir-root itself (default: {DEFAULT_CONFIGS})"
        ),
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])

    for config_name in args.configs:
        config_dir = args.model_dir_root / config_name if config_name else args.model_dir_root
        if not config_dir.is_dir():
            print(f"WARNING: {config_name or '(current)'}: directory not found, skipping: {config_dir}", file=sys.stderr)
            continue

        rows = []
        for exp_id in experiments:
            model_dir = config_dir / exp_id
            for fold in range(n_folds):
                best = load_weights(model_dir / f"{exp_id}.fold{fold}.torch")
                final = load_weights(model_dir / f"{exp_id}.fold{fold}.final.torch")
                if best is None and final is None:
                    continue
                rows.append(
                    {
                        "experiment": exp_id,
                        "fold": fold,
                        "best_lw0": best[0] if best else None,
                        "best_lw1": best[1] if best else None,
                        "best_frozen": best[2] if best else None,
                        "final_lw0": final[0] if final else None,
                        "final_lw1": final[1] if final else None,
                        "final_frozen": final[2] if final else None,
                    }
                )

        if not rows:
            print(f"\n=== {config_name or '(current)'}: no checkpoints found under {config_dir} ===", file=sys.stderr)
            continue

        df = pd.DataFrame(rows)
        df["best_w0_over_w1"] = (df["best_lw1"] ** 2) / (df["best_lw0"] ** 2)
        df["final_w0_over_w1"] = (df["final_lw1"] ** 2) / (df["final_lw0"] ** 2)

        def frozen_pct(col):
            # "frozen" is Python True/False/None (None when it can't be
            # recovered from a state_dict-style checkpoint); coerce to
            # float so a mix of those doesn't break .mean().
            frac = df[col].map({True: 1.0, False: 0.0}).mean()
            return "unknown (state_dict checkpoint)" if pd.isna(frac) else f"{100 * frac:.0f}%"

        print(f"\n=== {config_name or '(current)'}  (n={len(df)} fold checkpoints found) ===")
        print(
            "  best  (*.torch, what benchmark_cherimoya.py loads): "
            f"lw0 median={df['best_lw0'].median():.4f}  lw1 median={df['best_lw1'].median():.4f}  "
            f"w0/w1 median={df['best_w0_over_w1'].median():.4f}  "
            f"frozen at best epoch={frozen_pct('best_frozen')}"
        )
        print(
            "  final (*.final.torch, end of training):             "
            f"lw0 median={df['final_lw0'].median():.4f}  lw1 median={df['final_lw1'].median():.4f}  "
            f"w0/w1 median={df['final_w0_over_w1'].median():.4f}  "
            f"frozen by end={frozen_pct('final_frozen')}"
        )


if __name__ == "__main__":
    main()
