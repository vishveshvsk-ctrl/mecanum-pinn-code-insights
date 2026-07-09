#!/usr/bin/env python
"""Fleet-wide acceleration sidecar generation (see instructions/arrow-accel-augmentation.md).

For every original `<stem>.arrow` in a sweep directory, write a sidecar
`accel/<stem>_accel.arrow` holding the exact-dynamics accelerations
(dVx, dVy, dpsidot, dw1..4) computed from EXISTING stored columns via
`accel_dynamics.body_wheel_accels` (numpy port of the verified torch
`ne_rhs`). Originals are opened read-only and never rewritten. Idempotent
(valid sidecar present -> SKIPPED) and resumable via a streaming manifest CSV.

Usage:
    python make_accel_sidecars.py --dry-run
    python make_accel_sidecars.py --limit 10
    python make_accel_sidecars.py --workers 4
    python make_accel_sidecars.py --verify-only
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accel_dynamics as AD

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Simulation_Data_MecanumSlipSpin_LugreAdamov"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "sidecar_manifest.csv"
DEPRECATED_MARKERS = ("SimulationDataSlipSpin_Julia", "_mu_pilot2")

_FNAME_RE = re.compile(
    r'^(?P<profile>.+?)_c(?P<combo>\d+)_mu_(?P<mu>[0-9.]+)'
    r'_case(?P<fc>\d+)_(?P<fm>lugre_[a-z]+)_chi_(?P<chi>[0-9.]+)\.arrow$'
)

SOURCE_COLS = (
    ["Vx", "Vy", "psi_dot", "time"]
    + [f"w{i}" for i in range(1, 5)]
    + [f"Msat_{i}" for i in range(1, 5)]
    + [f"Fpar_{i}" for i in range(1, 5)]
    + [f"Fperp_{i}" for i in range(1, 5)]
)
SIDECAR_COLS = ["dVx", "dVy", "dpsidot", "dw1", "dw2", "dw3", "dw4"]

# Central-difference residual is diagnostic (dense-output truncation floor is
# unknown until measured on the pilot, per the brief); this only flags files
# for human review, it never fails verify_sidecar on its own.
DEFAULT_FD_REL_FLAG = 0.05


@dataclass
class SidecarConfig:
    data_dir: Path
    manifest_path: Path = DEFAULT_MANIFEST
    fd_rel_flag: float = DEFAULT_FD_REL_FLAG


@dataclass
class VerifyResult:
    ok: bool
    row_count_ok: bool
    finite_ok: bool
    metadata_ok: bool
    fd_max_residual: Dict[str, float] = field(default_factory=dict)
    fd_rms_residual: Dict[str, float] = field(default_factory=dict)
    fd_flagged: bool = False
    messages: List[str] = field(default_factory=list)


@dataclass
class FileResult:
    path: str
    status: str  # DONE | SKIPPED | REBUILT | FAILED
    wall_time: float
    rows: int = 0
    fd_max_residual: str = ""
    error: str = ""


def parse_arrow_filename(name: str) -> Optional[Dict[str, Any]]:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return {
        "profile": m["profile"], "combo": int(m["combo"]), "mu": float(m["mu"]),
        "fc": int(m["fc"]), "fm": m["fm"], "chi": float(m["chi"]),
    }


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime_ns)}"


def sidecar_path_for(orig: Path) -> Path:
    return orig.parent / "accel" / f"{orig.stem}_accel.arrow"


def _read_arrow(fp: Path, columns: Optional[List[str]] = None) -> "pa.Table":
    """Windows-safe Arrow read (mirrors mecanum_pinn.data's fallback pattern)."""
    kw: Dict[str, Any] = {}
    if sys.platform == "win32":
        kw["memory_map"] = False
    try:
        return feather.read_table(fp, columns=columns, **kw)
    except Exception:
        with pa.ipc.open_file(fp, memory_map=False) as reader:
            table = reader.read_all()
        return table.select(columns) if columns else table


def _write_metadata(schema: pa.Schema, meta: Dict[str, str]) -> pa.Schema:
    enc = {k.encode(): v.encode() for k, v in meta.items()}
    return schema.with_metadata(enc)


def _read_metadata(schema: pa.Schema) -> Dict[str, str]:
    raw = schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


REQUIRED_META_KEYS = (
    "source_name", "source_rows", "source_fingerprint",
    "eom_convention", "generator_version",
)


def _existing_sidecar_is_current(sidecar: Path, orig: Path) -> bool:
    """Cheap freshness check: open sidecar, compare metadata fingerprint only."""
    try:
        with pa.memory_map(str(sidecar), "r") as src:
            reader = pa.ipc.open_file(src)
            meta = _read_metadata(reader.schema)
    except Exception:
        return False
    if not all(k in meta for k in REQUIRED_META_KEYS):
        return False
    return (meta["source_name"] == orig.name
            and meta["source_fingerprint"] == fingerprint(orig))


def _central_fd(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    # The dense-output save grid has a handful of exact-duplicate timestamps
    # at solver/controller segment boundaries (dt=0) -> np.gradient divides by
    # zero there. Those points are masked out of the residual stats below.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(x, t, axis=0)


def _compute_verify(orig_cols: Dict[str, np.ndarray], sidecar_table: pa.Table,
                     orig: Path, cfg: SidecarConfig) -> VerifyResult:
    messages: List[str] = []
    meta = _read_metadata(sidecar_table.schema)

    metadata_ok = all(k in meta for k in REQUIRED_META_KEYS)
    if metadata_ok:
        metadata_ok = (meta["source_name"] == orig.name
                        and meta["source_fingerprint"] == fingerprint(orig)
                        and meta["eom_convention"] == AD.EOM_CONVENTION)
    if not metadata_ok:
        messages.append("metadata incomplete or fingerprint/convention mismatch")

    n_orig = orig_cols["Vx"].shape[0]
    row_count_ok = sidecar_table.num_rows == n_orig
    if not row_count_ok:
        messages.append(f"row count mismatch: sidecar={sidecar_table.num_rows} orig={n_orig}")
    if metadata_ok and int(meta["source_rows"]) != n_orig:
        row_count_ok = False
        messages.append("metadata source_rows disagrees with original row count")

    sc = {name: sidecar_table.column(name).to_numpy() for name in SIDECAR_COLS}
    finite_ok = all(np.isfinite(v).all() for v in sc.values())
    if not finite_ok:
        messages.append("non-finite values in sidecar")

    fd_max: Dict[str, float] = {}
    fd_rms: Dict[str, float] = {}
    fd_flagged = False
    if row_count_ok and finite_ok:
        t = orig_cols["time"]
        w = np.stack([orig_cols[f"w{i}"] for i in range(1, 5)], axis=-1)
        fd_targets = {
            "dVx": (_central_fd(orig_cols["Vx"], t), sc["dVx"]),
            "dVy": (_central_fd(orig_cols["Vy"], t), sc["dVy"]),
            "dpsidot": (_central_fd(orig_cols["psi_dot"], t), sc["dpsidot"]),
        }
        dw_fd = _central_fd(w, t)
        for i in range(4):
            fd_targets[f"dw{i+1}"] = (dw_fd[:, i], sc[f"dw{i+1}"])
        for name, (fd, exact) in fd_targets.items():
            # Interior points only: FD at the endpoints is one-sided and noisy.
            # A handful of interior points sit next to a duplicate (dt=0) save
            # timestamp -> np.gradient divides by zero there; excluded from stats.
            resid = np.abs(fd[1:-1].astype(np.float64) - exact[1:-1].astype(np.float64))
            finite = np.isfinite(resid)
            resid = resid[finite]
            fd_max[name] = float(resid.max()) if resid.size else 0.0
            fd_rms[name] = float(np.sqrt((resid ** 2).mean())) if resid.size else 0.0
            sig = exact[1:-1][finite].astype(np.float64)
            sig_rms = float(np.sqrt((sig ** 2).mean())) if sig.size else 0.0
            if sig_rms > 1e-9 and fd_rms[name] / sig_rms > cfg.fd_rel_flag:
                fd_flagged = True

    ok = row_count_ok and finite_ok and metadata_ok
    return VerifyResult(ok=ok, row_count_ok=row_count_ok, finite_ok=finite_ok,
                         metadata_ok=metadata_ok, fd_max_residual=fd_max,
                         fd_rms_residual=fd_rms, fd_flagged=fd_flagged, messages=messages)


def verify_sidecar(orig: Path, sidecar: Path, cfg: Optional[SidecarConfig] = None) -> VerifyResult:
    """Row count, finiteness, metadata/fingerprint pairing, FD cross-check residual."""
    cfg = cfg or SidecarConfig(data_dir=orig.parent)
    orig_table = _read_arrow(orig, columns=SOURCE_COLS)
    orig_cols = {name: orig_table.column(name).to_numpy() for name in SOURCE_COLS}
    sidecar_table = _read_arrow(sidecar)
    return _compute_verify(orig_cols, sidecar_table, orig, cfg)


def load_with_accel(orig: Path) -> pa.Table:
    """Reference join helper: original + its sidecar, alignment asserted via
    sidecar metadata (source_rows + fingerprint). Raises on mismatch/missing."""
    sidecar = sidecar_path_for(orig)
    if not sidecar.exists():
        raise FileNotFoundError(f"no sidecar for {orig.name}: expected {sidecar}")
    orig_table = _read_arrow(orig)
    sidecar_table = _read_arrow(sidecar)
    meta = _read_metadata(sidecar_table.schema)
    if not all(k in meta for k in REQUIRED_META_KEYS):
        raise ValueError(f"{sidecar} missing required metadata keys")
    if meta["source_name"] != orig.name:
        raise ValueError(f"{sidecar} paired with {meta['source_name']!r}, not {orig.name!r}")
    if meta["source_fingerprint"] != fingerprint(orig):
        raise ValueError(f"{sidecar} fingerprint stale vs current {orig.name}")
    if int(meta["source_rows"]) != orig_table.num_rows:
        raise ValueError(f"{sidecar} source_rows {meta['source_rows']} != orig rows {orig_table.num_rows}")
    if sidecar_table.num_rows != orig_table.num_rows:
        raise ValueError(f"{sidecar} row count {sidecar_table.num_rows} != orig rows {orig_table.num_rows}")
    out = orig_table
    for name in sidecar_table.column_names:
        out = out.append_column(name, sidecar_table.column(name))
    return out


def build_sidecar(path: Path, cfg: SidecarConfig, dry_run: bool = False) -> FileResult:
    """Idempotent sidecar generation; original opened read-only; tmp+replace write."""
    t0 = time.time()
    orig = Path(path)
    sidecar = sidecar_path_for(orig)
    accel_dir = sidecar.parent

    try:
        if sidecar.exists() and _existing_sidecar_is_current(sidecar, orig):
            return FileResult(path=str(orig), status="SKIPPED", wall_time=time.time() - t0)

        was_stale = sidecar.exists()
        if dry_run:
            return FileResult(path=str(orig), status="REBUILT" if was_stale else "DONE",
                               wall_time=time.time() - t0)

        parsed = parse_arrow_filename(orig.name)
        if parsed is None:
            return FileResult(path=str(orig), status="FAILED", wall_time=time.time() - t0,
                               error="filename does not match the Arrow naming contract")

        table = _read_arrow(orig, columns=SOURCE_COLS)
        n_rows = table.num_rows
        cols = {name: table.column(name).to_numpy() for name in SOURCE_COLS}
        w = np.stack([cols[f"w{i}"] for i in range(1, 5)], axis=-1)
        Msat = np.stack([cols[f"Msat_{i}"] for i in range(1, 5)], axis=-1)
        Fpar = np.stack([cols[f"Fpar_{i}"] for i in range(1, 5)], axis=-1)
        Fperp = np.stack([cols[f"Fperp_{i}"] for i in range(1, 5)], axis=-1)

        dVx, dVy, dpsidot, dw = AD.body_wheel_accels(
            cols["Vx"], cols["Vy"], cols["psi_dot"], w, Msat, Fpar, Fperp,
            friction_case=parsed["fc"])

        out_table = pa.table({
            "dVx": dVx, "dVy": dVy, "dpsidot": dpsidot,
            "dw1": dw[:, 0], "dw2": dw[:, 1], "dw3": dw[:, 2], "dw4": dw[:, 3],
        })
        meta = {
            "source_name": orig.name,
            "source_rows": str(n_rows),
            "source_fingerprint": fingerprint(orig),
            "eom_convention": AD.EOM_CONVENTION,
            "generator_version": AD.GENERATOR_VERSION,
        }
        out_table = out_table.replace_schema_metadata(
            {k.encode(): v.encode() for k, v in meta.items()})

        accel_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = sidecar.with_suffix(sidecar.suffix + ".tmp")
        with pa.OSFile(str(tmp_path), "wb") as sink:
            writer = pa.ipc.new_file(sink, out_table.schema)
            writer.write_table(out_table)
            writer.close()
        os.replace(tmp_path, sidecar)

        vr = _compute_verify(cols, out_table, orig, cfg)
        if not vr.ok:
            try:
                sidecar.unlink()
            except OSError:
                pass
            return FileResult(path=str(orig), status="FAILED", wall_time=time.time() - t0,
                               rows=n_rows, error="; ".join(vr.messages) or "verify failed")

        fd_summary = ",".join(f"{k}:{v:.3e}" for k, v in sorted(vr.fd_max_residual.items()))
        status = "REBUILT" if was_stale else "DONE"
        return FileResult(path=str(orig), status=status, wall_time=time.time() - t0,
                           rows=n_rows, fd_max_residual=fd_summary)
    except Exception as e:  # noqa: BLE001 - fleet job must never die on one file
        try:
            tmp_path = sidecar.with_suffix(sidecar.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return FileResult(path=str(orig), status="FAILED", wall_time=time.time() - t0, error=repr(e))


# ============================================================
# CLI orchestration
# ============================================================
def _check_disk_headroom(data_dir: Path, files: List[Path]) -> None:
    fleet_bytes = sum(p.stat().st_size for p in files)
    usage = shutil.disk_usage(str(data_dir))
    need = int(0.05 * fleet_bytes)
    if usage.free < need:
        raise SystemExit(
            f"[abort] free disk {usage.free/1e9:.1f} GB < 5% of fleet size "
            f"({fleet_bytes/1e9:.1f} GB -> need {need/1e9:.1f} GB headroom)")


def _enumerate(data_dir: Path, allow_deprecated: bool) -> List[Path]:
    if not allow_deprecated and any(m in str(data_dir) for m in DEPRECATED_MARKERS):
        raise SystemExit(f"[abort] {data_dir} looks like a DEPRECATED legacy dir; "
                          f"pass --allow-deprecated to override")
    files = sorted(p for p in data_dir.glob("*.arrow") if not p.name.endswith("_accel.arrow"))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N eligible files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="fleet-wide verify_sidecar re-run")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--fd-rel-flag", type=float, default=DEFAULT_FD_REL_FLAG)
    ap.add_argument("--allow-deprecated", action="store_true")
    args = ap.parse_args()

    workers = max(1, min(8, args.workers))
    cfg = SidecarConfig(data_dir=args.data_dir, manifest_path=args.manifest,
                         fd_rel_flag=args.fd_rel_flag)

    data_dir = args.data_dir.resolve()
    files = _enumerate(data_dir, args.allow_deprecated)
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[main] {len(files)} original .arrow files under {data_dir} "
          f"(workers={workers}, dry_run={args.dry_run}, verify_only={args.verify_only})")
    if not files:
        return

    if args.verify_only:
        _run_verify_only(files, cfg)
        return

    if not args.dry_run:
        (data_dir / "accel").mkdir(parents=True, exist_ok=True)
        _check_disk_headroom(data_dir, files)

    _run_build(files, cfg, workers, args.dry_run)


def _run_build(files: List[Path], cfg: SidecarConfig, workers: int, dry_run: bool) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # noqa: N816

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {"DONE": 0, "SKIPPED": 0, "REBUILT": 0, "FAILED": 0}
    worst_fd: List[str] = []
    total_bytes = 0

    with open(cfg.manifest_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "status", "wall_time", "rows", "fd_max_residual", "error"])

        iterator: Any
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(build_sidecar, fp, cfg, dry_run): fp for fp in files}
            iterator = as_completed(futs)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futs))
            for fut in iterator:
                r = fut.result()
                counts[r.status] = counts.get(r.status, 0) + 1
                w.writerow([r.path, r.status, f"{r.wall_time:.3f}", r.rows,
                            r.fd_max_residual, r.error])
                fh.flush()
                if r.status == "FAILED":
                    print(f"[FAILED] {r.path}: {r.error}")
                if r.fd_max_residual:
                    worst_fd.append(f"{Path(r.path).name}: {r.fd_max_residual}")
                if r.status in ("DONE", "REBUILT") and not dry_run:
                    sp = sidecar_path_for(Path(r.path))
                    if sp.exists():
                        total_bytes += sp.stat().st_size

    print(f"[summary] {counts}")
    print(f"[summary] sidecar bytes written: {total_bytes/1e9:.3f} GB")
    if worst_fd:
        print("[summary] sample FD max residuals (first 5):")
        for line in worst_fd[:5]:
            print(f"  {line}")


def _run_verify_only(files: List[Path], cfg: SidecarConfig) -> None:
    ok = 0
    bad: List[str] = []
    missing = 0
    for fp in files:
        sidecar = sidecar_path_for(fp)
        if not sidecar.exists():
            missing += 1
            continue
        vr = verify_sidecar(fp, sidecar, cfg)
        if vr.ok:
            ok += 1
        else:
            bad.append(f"{fp.name}: {'; '.join(vr.messages)}")
    print(f"[verify-only] ok={ok} bad={len(bad)} missing_sidecar={missing} total={len(files)}")
    for line in bad[:20]:
        print(f"  [bad] {line}")


if __name__ == "__main__":
    main()
