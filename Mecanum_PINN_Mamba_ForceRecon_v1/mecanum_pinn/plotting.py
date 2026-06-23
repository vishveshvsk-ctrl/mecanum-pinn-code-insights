"""Minimal static-matplotlib plotting (Agg backend, no interactive widgets).

Figures are saved under a per-run directory set by configure_figure_saving().
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

_SAVE_DIR = Path('.')
_RUN_TAG = ''


def configure_figure_saving(save_dir, run_tag: str = '') -> None:
    global _SAVE_DIR, _RUN_TAG
    _SAVE_DIR = Path(save_dir)
    _RUN_TAG = run_tag
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figs] saving to {_SAVE_DIR}")


def _save(fig, name: str) -> None:
    fig.savefig(_SAVE_DIR / f"{name}.png", dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_history(history: Dict, stage: str) -> None:
    h = history.get(stage, {})
    tr, va = h.get('train', {}), h.get('val', {})
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, hh, ls in (('train', tr, '-'), ('val', va, '--')):
        for key in ('total', 'grnd', 'phys', 'cons', 'param_id'):
            if key in hh and len(hh[key]):
                ax.plot(hh[key], ls, label=f"{split}/{key}", alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.set_title(f"{stage} loss history ({_RUN_TAG})")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, f"history_{stage}")
