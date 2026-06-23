#!/usr/bin/env python
# =============================================================================
# dataset_chunker.py — TEMPORARILY split the flat lugre_adamov .arrow dataset
# into <= N GB subfolders for transfer to another machine, then put it back.
#
# Lossless + verified via a manifest (chunk_manifest.csv at the data root). Moves
# are os.rename within the same volume = instant metadata ops (no data copy),
# even for ~209 GB. ONLY *.arrow files are moved; the percentile CSV, job logs,
# and the manifest stay at the root. Reassembly restores the exact flat layout.
#
# Run from code_insights/ :
#   python dataset_chunker.py --mode plan                  # read-only: show the chunk plan
#   python dataset_chunker.py --mode chunk                 # move *.arrow into chunk_000/...
#   python dataset_chunker.py --mode reassemble            # move them all back, remove chunk dirs
#
# NOTE: this folder is under OneDrive — the moves are local+instant, but OneDrive
# will sync the new structure afterward. Pause OneDrive sync first if you don't
# want that churn while transferring.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import os
import stat
from pathlib import Path

DEFAULT_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"


def list_arrows(d: Path):
    return sorted(d.glob("*.arrow"))


def plan_chunks(files, cap_bytes):
    """Greedy bin-pack (deterministic, sorted by name) into <= cap_bytes chunks."""
    chunks, cur, cur_sz = [], [], 0
    for f in files:
        sz = f.stat().st_size
        if sz > cap_bytes:
            raise SystemExit(f"single file {f.name} ({sz/1e9:.2f} GB) exceeds the cap")
        if cur and cur_sz + sz > cap_bytes:
            chunks.append(cur); cur, cur_sz = [], 0
        cur.append((f, sz)); cur_sz += sz
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser(description="Chunk / reassemble the lugre_adamov dataset.")
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--chunk-gb", type=float, default=9.0, help="max chunk size in decimal GB")
    ap.add_argument("--mode", choices=["plan", "chunk", "reassemble"], required=True)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    dd = Path(a.data_dir)
    cap = int(a.chunk_gb * 1000 ** 3)
    manifest = Path(a.manifest) if a.manifest else dd / "chunk_manifest.csv"

    if a.mode in ("plan", "chunk"):
        files = list_arrows(dd)
        if not files:
            raise SystemExit(f"no *.arrow at {dd} root (already chunked? use --mode reassemble)")
        chunks = plan_chunks(files, cap)
        total = sum(sz for ch in chunks for _, sz in ch)
        print(f"[{a.mode}] {len(files)} files, {total/1e9:.1f} GB -> {len(chunks)} chunks "
              f"(cap {a.chunk_gb} GB each)")
        for i, ch in enumerate(chunks):
            print(f"  chunk_{i:03d}: {len(ch):4d} files, {sum(sz for _, sz in ch)/1e9:6.2f} GB")
        if a.mode == "plan":
            print("[plan] read-only — nothing moved.")
            return
        rows = []
        for i, ch in enumerate(chunks):
            cdir = dd / f"chunk_{i:03d}"; cdir.mkdir(exist_ok=True)
            for f, sz in ch:
                os.rename(f, cdir / f.name)
                rows.append(dict(file=f.name, chunk=f"chunk_{i:03d}", size_bytes=sz))
        with open(manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "chunk", "size_bytes"])
            w.writeheader(); w.writerows(rows)
        left = len(list_arrows(dd))
        moved = sum(len(list((dd / f"chunk_{i:03d}").glob("*.arrow"))) for i in range(len(chunks)))
        print(f"[chunk] moved {moved} files into {len(chunks)} chunks; "
              f"{left} .arrow remain at root; manifest -> {manifest}")
        assert moved == len(rows) and left == 0, "VERIFY FAILED — check the folder!"
        print("[chunk] verified OK.")

    else:  # reassemble
        if not manifest.exists():
            raise SystemExit(f"manifest {manifest} not found — cannot reassemble safely")
        rows = list(csv.DictReader(open(manifest)))
        moved, missing = 0, []
        for r in rows:
            src = dd / r["chunk"] / r["file"]; dst = dd / r["file"]
            if src.exists():
                os.rename(src, dst); moved += 1
            elif dst.exists():
                moved += 1                                   # already restored
            else:
                missing.append(r["file"])
        for cdir in sorted(dd.glob("chunk_*")):
            if not (cdir.is_dir() and cdir.name[6:].isdigit()):
                continue                                     # skip chunk_manifest.csv etc.
            rem = list(cdir.iterdir())
            if rem:
                print(f"[reassemble] WARNING {cdir.name} still has {len(rem)} files — left in place")
                continue
            try:                                             # OneDrive placeholders are readonly
                os.chmod(cdir, stat.S_IWRITE)                # reparse points -> clear readonly first
                cdir.rmdir()
            except Exception as e:                           # never abort over an empty-dir cleanup
                print(f"[reassemble] couldn't remove empty {cdir.name}: {e!r} (delete manually if needed)")
        root_n = len(list_arrows(dd))
        print(f"[reassemble] restored {moved}/{len(rows)} files; {len(missing)} missing; "
              f"root now has {root_n} .arrow")
        if missing:
            print("  missing (first 10):", missing[:10])
        else:
            print("[reassemble] verified OK — manifest can be deleted once you've confirmed.")


if __name__ == "__main__":
    main()
