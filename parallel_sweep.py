#!/usr/bin/env python
# =============================================================================
# parallel_sweep.py — shared, approach-agnostic parallel-execution core.
#
# ONE launcher idiom for BOTH the A1 force-reconstruction PINN
# (Mecanum_PINN_Mamba_ForceRecon_v1/) and the A2 state observer (observer_v1_py/).
# `N = max_parallel` (the degree of parallelism) is the only machine-dependent
# knob; everything else is identical across approaches and across the 6 GB laptop
# / 24 GB workstation. Each package ships a thin `launch_parallel.py` that builds
# a list of `Job`s + (optionally) warms the decimated cache, then calls run_sweep.
#
# This module is a PURE PROCESS ORCHESTRATOR — it never imports torch, numpy, or
# either training package. It only:
#   * caps concurrency at N (process pool, free-slot scheduler),
#   * (Linux) pins each job to a disjoint CPU-core block via taskset so the
#     dataloaders don't contend,
#   * tees each job's stdout/stderr to its own log,
#   * is RESUME-SAFE: a job whose run_dir/metrics.json already exists is skipped
#     (--force overrides),
#   * harvests every run's metrics.json into one ranking CSV (universal columns
#     + the union of whatever metric keys each approach writes).
#
# The decimated-cache half of the pattern lives in each package's data.py
# (read_arrays / read_trajectory + warm_cache); --warm-cache in the per-package
# launcher runs ONE single-process pre-build pass before fan-out so N jobs never
# race to write the same .npz.
# =============================================================================
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent                  # code_insights/


@dataclass
class Job:
    """One independent training run.

    label    : unique; names the per-job log and (by convention) the run_tag.
    pkg_dir  : package dir put on PYTHONPATH (rel to code_insights/), e.g.
               "observer_v1_py" or "Mecanum_PINN_Mamba_ForceRecon_v1".
    entry    : worker script (rel to code_insights/), e.g.
               "observer_v1_py/train_observer.py".
    args     : CLI tokens passed to the worker (all str).
    run_dir  : dir (rel to code_insights/) where the worker writes metrics.json;
               used for resume-skip AND for the CSV harvest. None => no skip/harvest.
    """
    label: str
    pkg_dir: str
    entry: str
    args: List[str] = field(default_factory=list)
    run_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# CPU affinity (Linux only) + environment
# ---------------------------------------------------------------------------
def core_block(slot: int, cores_per_job: int, ncpu: int) -> str:
    lo = (slot * cores_per_job) % max(ncpu, 1)
    return f"{lo}-{min(lo + cores_per_job - 1, ncpu - 1)}"


def _ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0))             # Linux
    except Exception:
        return os.cpu_count() or 8


def _env_for(job: Job) -> Dict[str, str]:
    env = os.environ.copy()
    if job.pkg_dir:                                     # script dir already lands on
        prev = env.get("PYTHONPATH", "")                # sys.path[0]; this is belt-and-
        pkg = str(ROOT / job.pkg_dir)                   # suspenders for `import <pkg>`.
        env["PYTHONPATH"] = pkg + (os.pathsep + prev if prev else "")
    return env


def is_done(job: Job) -> bool:
    """A run is COMPLETE iff it has written its run_dir/metrics.json. (A2 also
    epoch-resumes from checkpoint.pt; A1 restarts an interrupted run from scratch.
    Either way, only the metrics.json marker means 'don't bother re-launching'.)"""
    return bool(job.run_dir) and (ROOT / job.run_dir / "metrics.json").exists()


def _full_cmd(job: Job, python: str, slot: int, cores_per_job: int,
              ncpu: int, use_taskset: bool) -> List[str]:
    # -u = unbuffered stdout/stderr: when the launcher redirects a worker to a
    # log file (non-TTY), Python otherwise BLOCK-buffers prints, so per-epoch
    # progress wouldn't reach the log (or the status heartbeat) until the buffer
    # fills or the process exits. -u makes the live monitoring actually live.
    base = [python, "-u", job.entry] + [str(a) for a in job.args]
    if use_taskset:
        return ["taskset", "-c", core_block(slot, cores_per_job, ncpu)] + base
    return base


# ---------------------------------------------------------------------------
# Single blocking run (used for the --warm-cache pre-build pass)
# ---------------------------------------------------------------------------
def run_blocking(label: str, pkg_dir: str, entry: str, args: List[str],
                 python: str, log_path: Path) -> int:
    job = Job(label=label, pkg_dir=pkg_dir, entry=entry, args=args)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] warm-cache pass (single process): {label}  -> {log_path}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        rc = subprocess.call(_full_cmd(job, python, 0, 1, _ncpu(), False),
                             cwd=str(ROOT), env=_env_for(job),
                             stdout=fh, stderr=subprocess.STDOUT)
    print(f"[sweep] warm-cache {'done' if rc == 0 else 'FAILED'} "
          f"(rc={rc}, {(time.time()-t0)/60:.1f} min)")
    return rc


# ---------------------------------------------------------------------------
# Metric harvest -> one ranking CSV
# ---------------------------------------------------------------------------
def harvest(jobs: List[Job], csv_path: Path,
            status: Optional[Dict[str, Tuple[str, object, float]]] = None) -> None:
    status = status or {}
    rows: List[Dict[str, object]] = []
    metric_keys: List[str] = []                          # preserve first-seen order
    for j in jobs:
        st, rc, wall = status.get(j.label, ("?", "", 0.0))
        row: Dict[str, object] = {
            "label": j.label, "pkg": j.pkg_dir, "status": st, "rc": rc,
            "wall_min": (f"{wall/60:.2f}" if isinstance(wall, (int, float)) and wall else ""),
            "run_dir": j.run_dir or "",
        }
        mp = (ROOT / j.run_dir / "metrics.json") if j.run_dir else None
        if mp and mp.exists():
            try:
                with open(mp) as fh:
                    d = json.load(fh)
            except Exception as e:
                print(f"[sweep] harvest: bad metrics.json for {j.label}: {e!r}")
                d = {}
            for k, v in d.items():
                k = str(k)
                if k not in metric_keys:
                    metric_keys.append(k)
                row[k] = v
        rows.append(row)
    cols = ["label", "pkg", "status", "rc", "wall_min", "run_dir"] + metric_keys
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[sweep] harvested {len(rows)} runs ({len(metric_keys)} metric cols) -> {csv_path}")


# ---------------------------------------------------------------------------
# Live progress (for unattended runs): tail each running job's log for its
# current epoch, emit a heartbeat + a single at-a-glance status file.
# ---------------------------------------------------------------------------
def _tail_line(path: Path, nbytes: int = 8192) -> str:
    """Last non-empty line of a log (reads only the tail; strips tqdm '\\r')."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            data = fh.read().decode("utf-8", "replace")
    except Exception:
        return ""
    for line in reversed(data.splitlines()):
        seg = line.split("\r")[-1].strip()          # tqdm rewrites with \r
        if seg:
            return seg[:140]
    return ""


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------
def run_sweep(jobs: List[Job], *, max_parallel: int, python: str,
              log_dir: str, cores_per_job: int = 4,
              csv_path: Optional[str] = None, dry_run: bool = False,
              force: bool = False, mps_note: bool = True,
              poll_seconds: float = 3.0,
              heartbeat_seconds: float = 120.0) -> Dict[str, List[str]]:
    """Run `jobs` with at most `max_parallel` concurrent worker processes.

    Returns {"done":[labels], "failed":[labels], "skipped":[labels]}.
    """
    ncpu = _ncpu()
    use_taskset = platform.system() == "Linux" and bool(shutil.which("taskset"))
    pending = [j for j in jobs if force or not is_done(j)]
    skipped = [j for j in jobs if not force and is_done(j)]

    print(f"[sweep] {len(jobs)} jobs | {len(pending)} to run | "
          f"{len(skipped)} already done | max_parallel={max_parallel} "
          f"(cpus={ncpu}, taskset={'on' if use_taskset else 'off'})")
    if skipped:
        names = ", ".join(j.label for j in skipped[:8]) + (" ..." if len(skipped) > 8 else "")
        print(f"[sweep] resume-skip (metrics.json present): {names}")
    if mps_note and not use_taskset:
        print("[sweep] tip: on Linux, start NVIDIA MPS for true concurrent GPU "
              "kernels (`nvidia-cuda-mps-control -d` before; "
              "`echo quit | nvidia-cuda-mps-control` after). Windows uses time-slicing.")

    if dry_run:
        for i, j in enumerate(pending):
            cmd = _full_cmd(j, python, i % max(max_parallel, 1),
                            cores_per_job, ncpu, use_taskset)
            print(f"  [{i:2d}] {j.label}: PYTHONPATH={j.pkg_dir} {' '.join(cmd)}")
        print(f"[sweep] dry-run: {len(pending)} would run, {len(skipped)} skipped")
        return {"done": [], "failed": [], "skipped": [j.label for j in skipped]}

    log_path = ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    status_path = log_path / "sweep_status.txt"
    queue = list(pending)
    running: List[list] = []                             # [proc, job, fh, slot, t0]
    free = list(range(max_parallel))
    done: List[Tuple[Job, int, float]] = []
    failed: List[Tuple[Job, int, float]] = []
    t_start = time.time()
    last_hb = t_start - heartbeat_seconds                # -> emit once on the first poll

    def emit_status(to_terminal: bool) -> None:
        now = time.time()
        head = (f"[sweep] {(now - t_start) / 60:.0f} min | done {len(done)} "
                f"failed {len(failed)} running {len(running)} queued {len(queue)} "
                f"/ {len(pending)} to run")
        runlines = [f"  - {jb.label} ({(now - t0) / 60:.0f}m): "
                    f"{_tail_line(log_path / (jb.label + '.log'))}"
                    for _p, jb, _fh, _sl, t0 in running]
        body = [head] + runlines
        if done:
            body.append("done:   " + ", ".join(j.label for j, _, _ in done))
        if failed:
            body.append("FAILED: " + ", ".join(j.label for j, _, _ in failed))
        if queue:
            body.append("queued: " + ", ".join(j.label for j in queue[:16])
                        + (" ..." if len(queue) > 16 else ""))
        try:
            status_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        except Exception:
            pass
        if to_terminal:
            try:                          # a display/encoding hiccup (e.g. a non-ASCII
                print(head)               # char tailed from a worker log printed to a
                for rl in runlines:       # cp1252 stream) must NEVER kill the orchestrator
                    print(rl)
            except Exception:
                pass

    print(f"[sweep] live status -> {status_path}")
    print(f"[sweep] per-job logs -> {log_path}/<label>.log  "
          f"(watch a run: tail -f, or PowerShell `Get-Content <log> -Wait`)")

    while queue or running:
        while queue and free:
            slot = free.pop(0)
            job = queue.pop(0)
            fh = open(log_path / f"{job.label}.log", "w")
            cmd = _full_cmd(job, python, slot, cores_per_job, ncpu, use_taskset)
            p = subprocess.Popen(cmd, cwd=str(ROOT), env=_env_for(job),
                                 stdout=fh, stderr=subprocess.STDOUT)
            running.append([p, job, fh, slot, time.time()])
            print(f"[sweep] start {job.label} (slot {slot}) -> "
                  f"{log_path / (job.label + '.log')}")
        time.sleep(poll_seconds)
        for rec in list(running):
            p, job, fh, slot, t0 = rec
            if p.poll() is not None:
                fh.close()
                running.remove(rec)
                free.append(slot)
                wall = time.time() - t0
                (done if p.returncode == 0 else failed).append((job, p.returncode, wall))
                print(f"[sweep] {'done' if p.returncode == 0 else 'FAILED'} "
                      f"{job.label} (rc={p.returncode}, {wall/60:.1f} min) "
                      f"[{len(done)+len(failed)}/{len(pending)}]")
        now = time.time()
        if now - last_hb >= heartbeat_seconds:           # periodic heartbeat + status file
            last_hb = now
            emit_status(to_terminal=True)

    emit_status(to_terminal=False)                       # final snapshot
    print(f"\n[sweep] complete: {len(done)} ok, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if failed:
        print("  failed:", ", ".join(j.label for j, _, _ in failed))

    if csv_path:
        status: Dict[str, Tuple[str, object, float]] = {}
        for j, rc, w in done:
            status[j.label] = ("ok", rc, w)
        for j, rc, w in failed:
            status[j.label] = ("FAILED", rc, w)
        for j in skipped:
            status[j.label] = ("skipped", 0, 0.0)
        harvest(jobs, ROOT / csv_path, status=status)

    return {"done": [j.label for j, _, _ in done],
            "failed": [j.label for j, _, _ in failed],
            "skipped": [j.label for j in skipped]}
