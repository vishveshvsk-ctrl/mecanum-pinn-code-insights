"""Training manifest (JSON) — records the exact split + scope + knobs next to
the checkpoints, so figures/eval can reconstruct the test set deterministically.
JSON (not TOML) to avoid a writer dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def save_training_manifest(*, ckpt_dir, run_tag: str,
                           profiles, mu_values, chi_values,
                           data_dir, whitelist_path,
                           seed: int,
                           train_names: Sequence[str],
                           val_names: Sequence[str],
                           test_names: Sequence[str],
                           config_summary: Dict[str, Any],
                           stages_trained: Sequence[str],
                           forward_ckpt_ref: str = "",
                           regime: str = "") -> None:
    out_dir = Path(ckpt_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_tag': run_tag,
        'regime': str(regime),
        'scope': {'profiles': list(profiles),
                  'mu_values': list(mu_values),
                  'chi_values': list(chi_values),
                  'data_dir': str(data_dir),
                  'whitelist_path': str(whitelist_path),
                  'seed': seed},
        'split': {'train_names': list(train_names),
                  'val_names': list(val_names),
                  'test_names': list(test_names)},
        'config_summary': config_summary,
        'stages_trained': list(stages_trained),
        'forward_ckpt_ref': forward_ckpt_ref,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(payload, indent=2))
    print(f"[manifest] wrote {out_dir / 'manifest.json'}")


def load_manifest_at(ckpt_path) -> Optional[Dict[str, Any]]:
    m = Path(ckpt_path).parent / 'manifest.json'
    if m.exists():
        return json.loads(m.read_text())
    return None
