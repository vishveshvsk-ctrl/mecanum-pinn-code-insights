"""Figure-save infrastructure + generic history / comparison plots.

The module-level state (_FIG_DIR, _FIG_RUN_TAG, _FIG_COUNTER) is set by
configure_figure_saving() at the start of a run; every plot helper calls
save_figure() instead of plt.show(), which writes a numbered PNG file
with the run_tag baked into the filename.

Topic-specific plots (test-trajectory predictions, mu/chi rolling mean)
live in evaluation.py and trajectory_eval.py and use save_figure from here.
"""
from __future__ import annotations

import os as _os_for_mpl
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

# Headless detection: if no DISPLAY on Linux (and not Windows), force Agg
# so SSH / nohup / detached runs don't try to open an X11 display.
if _os_for_mpl.environ.get('DISPLAY') is None and _os_for_mpl.name != 'nt':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt   # noqa: E402  (must come after backend choice)


# ============================================================
# Module state
# ============================================================
_FIG_COUNTER: int           = 0
_FIG_DIR:     Optional[Path] = None
_FIG_RUN_TAG: Optional[str]  = None


def configure_figure_saving(fig_dir: Path, run_tag: str) -> None:
    """Set the figure output directory and run tag. Resets the counter so
    re-runs of main() produce a fresh figure_01..N sequence."""
    global _FIG_COUNTER, _FIG_DIR, _FIG_RUN_TAG
    _FIG_DIR     = Path(fig_dir)
    _FIG_RUN_TAG = run_tag
    _FIG_COUNTER = 0
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figs] saving to {_FIG_DIR.resolve()} (filenames: figure_NN_<run_tag>.png)")


def get_fig_counter() -> int:
    return _FIG_COUNTER


def save_figure(label: str = "", dpi: int = 120) -> Optional[Path]:
    """Save the current matplotlib figure as figure_NN_<run_tag>[_label].png
    into the configured figure directory, then close it.

    If configure_figure_saving() was never called, falls back to plt.show()
    so notebook-style use still works.
    """
    global _FIG_COUNTER
    if _FIG_DIR is None or _FIG_RUN_TAG is None:
        plt.show()
        return None

    _FIG_COUNTER += 1
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") if label else ""
    if safe_label:
        fname = f"figure_{_FIG_COUNTER:02d}_{_FIG_RUN_TAG}_{safe_label}.png"
    else:
        fname = f"figure_{_FIG_COUNTER:02d}_{_FIG_RUN_TAG}.png"
    out = _FIG_DIR / fname

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [figs] saved {out.name}")
    return out


# ============================================================
# Generic history + comparison plots
# ============================================================
def plot_history(history: Dict, stage: str = 'forward'):
    """Per-component log-y train+val curves for a single stage."""
    h = history[stage]
    keys = [k for k in ('total', 'state', 'grnd', 'phys', 'cons')
            if k in h['train']]
    fig, axs = plt.subplots(len(keys), 1, figsize=(10, 2.2 * len(keys)), sharex=True)
    if len(keys) == 1:
        axs = [axs]
    for ax, k in zip(axs, keys):
        ax.plot(h['train'][k], label='train', alpha=0.8)
        if k in h['val']:
            ax.plot(h['val'][k],   label='val',   alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylabel(k)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel('epoch')
    fig.suptitle(f"{stage} stage history")
    save_figure(label=f"{stage}_history")


def plot_train_val_test_comparison(history: Dict, test_metrics: Dict,
                                   stage: str, figsize=(10, 5)):
    """Grouped bar chart of final-epoch train, final-epoch val, and test losses."""
    h = history[stage]
    candidates = ['total', 'state', 'grnd', 'phys', 'cons']
    keys = [k for k in candidates
            if k in test_metrics
            and k in h['train'] and len(h['train'][k]) > 0]
    if not keys:
        print("[plot] no comparable metrics to draw")
        return

    def _last(d, k):
        v = d.get(k, [])
        return v[-1] if len(v) > 0 else float('nan')

    train_vals = [_last(h['train'], k) for k in keys]
    val_vals   = [_last(h['val'],   k) for k in keys]
    test_vals  = [test_metrics[k]      for k in keys]

    x = np.arange(len(keys)); w = 0.27
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w, train_vals, w, label='train (final)', color='#4c72b0')
    ax.bar(x,     val_vals,   w, label='val (final)',   color='#dd8452')
    ax.bar(x + w, test_vals,  w, label='test',          color='#55a868')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel('loss (log scale)')
    ax.set_title(f'Train / val / test — stage={stage}', fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, which='both')
    ax.legend()
    save_figure(label=f"{stage}_train_val_test")

    print(f"\n{stage.upper()} stage — final train/val/test:")
    print(f"{'component':<10}{'train':>14}{'val':>14}{'test':>14}")
    for k, t, v, e in zip(keys, train_vals, val_vals, test_vals):
        print(f"{k:<10}{t:>14.3e}{v:>14.3e}{e:>14.3e}")


def plot_id_vs_ood_comparison(id_metrics: Dict[str, float],
                              ood_metrics: Dict[str, float],
                              stage: str, label_ood: str = 'OOD',
                              figsize=(10, 5)):
    """Grouped bar chart, log-y, with degradation ratios printed."""
    candidates = ['total', 'state', 'grnd', 'phys', 'cons']
    keys = [k for k in candidates if k in id_metrics and k in ood_metrics]
    if not keys:
        print("[plot] no comparable metrics")
        return

    id_vals  = [id_metrics[k]  for k in keys]
    ood_vals = [ood_metrics[k] for k in keys]
    x = np.arange(len(keys)); w = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w/2, id_vals,  w, label='in-distribution', color='#55a868')
    ax.bar(x + w/2, ood_vals, w, label=label_ood,         color='#c44e52')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel('loss (log scale)')
    ax.set_title(f'In-distribution vs {label_ood} — stage={stage}', fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, which='both')
    ax.legend()

    print(f"\n{stage.upper()} stage — {label_ood} ratio:")
    print(f"{'component':<10}{'in-dist':>14}{label_ood:>14}{'ratio':>10}")
    for k, a, b in zip(keys, id_vals, ood_vals):
        ratio = b / a if a > 0 else float('inf')
        print(f"{k:<10}{a:>14.3e}{b:>14.3e}{ratio:>10.2f}x")
    save_figure(label=f"ood_{stage}_{label_ood}")
