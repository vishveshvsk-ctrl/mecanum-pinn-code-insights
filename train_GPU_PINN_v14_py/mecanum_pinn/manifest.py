"""Training manifest -- captures exactly what scope/split a checkpoint represents.

Saved as `manifest.toml` next to the .pth files in each checkpoint folder. Read
by the figures stage and by plot_ood.py so they can reconstruct the in-distribution
test set without depending on a re-run of stratified_split (which is fragile if
the whitelist file changes between training and evaluation).

Format chosen for human readability -- you can `cat manifest.toml` and see
exactly which trajectories went into train / val / test.

Writer: hand-rolled, no extra deps. Schema is two levels max so a small emitter
suffices.
Reader: tomllib (Python 3.11+) with tomli fallback for older Pythons.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --- read backend -----------------------------------------------------------
try:
    import tomllib                       # Python 3.11+
except ImportError:                      # pragma: no cover
    try:
        import tomli as tomllib          # type: ignore
    except ImportError as e:             # pragma: no cover
        raise ImportError(
            "Reading manifests needs Python 3.11+ (built-in tomllib) or "
            "the 'tomli' package on older Pythons. pip install tomli"
        ) from e

MANIFEST_FILENAME = "manifest.toml"
SCHEMA_VERSION    = 1


# ============================================================
# TOML emitter (writer side)
# ============================================================
def _emit_str(s: str) -> str:
    """Emit a TOML string. Use literal (single-quoted) form for paths / strings
    that contain backslashes -- avoids the visual mess of escaping every \\.
    Falls back to basic (double-quoted) form when the string contains a single
    quote or a newline."""
    if ("\\" in s or "/" in s) and ("'" not in s) and ("\n" not in s):
        return f"'{s}'"
    s_esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s_esc}"'


def _emit_value(v: Any) -> str:
    if isinstance(v, bool):                 # check before int
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _emit_str(v)
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        items = [_emit_value(x) for x in v]
        # Inline short lists; multi-line for long ones (e.g. trajectory names)
        if len(v) <= 6 and sum(len(s) for s in items) < 80:
            return "[" + ", ".join(items) + "]"
        body = ",\n    ".join(items)
        return "[\n    " + body + ",\n]"
    raise TypeError(f"can't emit TOML value of type {type(v).__name__}: {v!r}")


def _emit_section(name: str, body: Dict[str, Any]) -> List[str]:
    out = [f"[{name}]"]
    for k, v in body.items():
        out.append(f"{k} = {_emit_value(v)}")
    return out


def write_toml(data: Dict[str, Any], path: Path) -> None:
    """Emit a dict as TOML. Schema: top-level scalars + one level of [section]
    tables. Don't pass nested-of-nested -- we'd need a more capable emitter."""
    lines: List[str] = []
    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_emit_value(v)}")
    if any(not isinstance(v, dict) for v in data.values()):
        lines.append("")
    for k, v in data.items():
        if isinstance(v, dict):
            lines.extend(_emit_section(k, v))
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def read_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ============================================================
# Manifest API
# ============================================================
def save_training_manifest(
    ckpt_dir: Path,
    run_tag: str,
    *,
    motion_cases:          Sequence[str],
    mu_values:             Sequence[float],
    chi_values:            Sequence[float],
    data_dir:              Path,
    whitelist_path:        Path,
    whitelist_total_count: int,
    subsample_n:           int,
    seed:                  int,
    train_names:           Sequence[str],
    val_names:             Sequence[str],
    test_names:            Sequence[str],
    config_summary:        Dict[str, Any],
    stages_trained:        Sequence[str],
    forward_ckpt_ref:      str = "",
) -> Path:
    """Write manifest.toml into ckpt_dir / run_tag /.

    Parameters
    ----------
    ckpt_dir, run_tag : output folder = ckpt_dir / run_tag
    motion_cases, mu_values, chi_values : training scope (what was IN training)
    data_dir, whitelist_path : data provenance
    whitelist_total_count    : entries in the whitelist .txt file
    subsample_n              : K passed to parse_whitelist (0 = no subsample)
    seed, train/val/test_names : split definition (the actual lists, by file name)
    config_summary  : architectural/runtime knobs (seq_len, hidden_dim, etc)
    stages_trained  : ['forward'] | ['inverse_H'] | ['inverse_NoH'] | combinations
    forward_ckpt_ref : when this is an inverse run, the forward .pth that was
                       loaded as the backbone. Empty for forward-only runs.

    Returns
    -------
    Path to the written manifest.toml.
    """
    out_dir = Path(ckpt_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / MANIFEST_FILENAME

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at":     _dt.datetime.now(_dt.timezone.utc)
                              .isoformat(timespec="seconds"),
        "stages": {
            "trained":          list(stages_trained),
            "forward_ckpt_ref": forward_ckpt_ref,
        },
        "scope": {
            "motion_cases": list(motion_cases),
            "mu_values":    [float(v) for v in mu_values],
            "chi_values":   [float(v) for v in chi_values],
        },
        "data_source": {
            "data_dir":              str(data_dir),
            "whitelist_path":        str(whitelist_path),
            "whitelist_total_count": int(whitelist_total_count),
            "subsample_n":           int(subsample_n),
        },
        "split": {
            "seed":         int(seed),
            "train_count":  len(train_names),
            "val_count":    len(val_names),
            "test_count":   len(test_names),
            "train_names":  list(train_names),
            "val_names":    list(val_names),
            "test_names":   list(test_names),
        },
        "config_summary": dict(config_summary),
    }
    write_toml(payload, out)
    print(f"[manifest] wrote {out}  "
          f"(train={len(train_names)} val={len(val_names)} test={len(test_names)})")
    return out


def load_training_manifest(ckpt_dir: Path, run_tag: str
                           ) -> Optional[Dict[str, Any]]:
    """Read manifest.toml from ckpt_dir / run_tag /. Returns None if missing."""
    p = Path(ckpt_dir) / run_tag / MANIFEST_FILENAME
    if not p.exists():
        return None
    return read_toml(p)


def load_manifest_at(path: Path) -> Optional[Dict[str, Any]]:
    """Read manifest.toml relative to a flexible path argument.

    Accepts:
      - a directory path -> look for manifest.toml inside it
      - a .pth file path -> look for manifest.toml in the same directory
      - the manifest path itself -> read it directly
    Returns the parsed dict, or None if the manifest is missing.
    """
    p = Path(path)
    if p.is_file() and p.name == MANIFEST_FILENAME:
        return read_toml(p)
    if p.is_dir():
        m = p / MANIFEST_FILENAME
        return read_toml(m) if m.exists() else None
    if p.is_file() and p.suffix == ".pth":
        m = p.parent / MANIFEST_FILENAME
        return read_toml(m) if m.exists() else None
    return None
