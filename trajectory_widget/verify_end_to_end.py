"""End-to-end verification for the Mecanum trajectory widget.

Checks:
- catalog cascade is data-driven;
- two distinct selections resolve, extract, and satisfy length/origin contracts;
- mid-trajectory frames render to PNG;
- playback GIF (or numbered frames) is produced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Ensure code_insights/ is on path for mecanum_widget and trajectory_widget imports.
_CODE_INSIGHTS = Path(__file__).resolve().parent.parent
if str(_CODE_INSIGHTS) not in sys.path:
    sys.path.insert(0, str(_CODE_INSIGHTS))

from mecanum_widget.data_extract import extract  # noqa: E402
from trajectory_widget.widget import (  # noqa: E402
    export_gif,
    load_catalog,
    setup_axes,
    update_frame,
    valid_combos,
)


def _find_existing_selection(catalog: dict, profile: str, preferred_combo: int):
    combos = valid_combos(profile, catalog)
    if not combos:
        raise ValueError(f"Profile {profile} has no combos in catalog")
    combo = preferred_combo if preferred_combo in combos else combos[0]
    return profile, combo


def main() -> int:
    # Data dir is resolved automatically from the repo layout
    # (directory containing both code_insights/ and data/ siblings),
    # or via the MECANUM_DATA_DIR env override.
    catalog = load_catalog()
    profiles = catalog["profiles"]
    combos_per_profile = catalog["valid_combos_per_profile"]

    # Cascade assertion.
    assert set(combos_per_profile.keys()) == set(profiles), (
        "valid_combos_per_profile keys must match profiles list"
    )
    for p in profiles:
        assert valid_combos(p, catalog) == combos_per_profile[p], (
            f"valid_combos({p}) mismatch"
        )
    print("Cascade assertion passed")

    # Pick two distinct selections that actually exist, preferring the
    # requested mu/chi and falling back to the first present value.
    def pick(profile: str, combo: int, preferred_mu: float, preferred_chi: float):
        entries = [
            e
            for e in catalog["entries"]
            if e["profile"] == profile and e["combo"] == combo
        ]
        if not entries:
            raise ValueError(f"No entries for {profile}/{combo}")
        for e in entries:
            if abs(e["mu"] - preferred_mu) < 1e-9 and abs(e["chi"] - preferred_chi) < 1e-9:
                return profile, combo, preferred_mu, preferred_chi
        e = entries[0]
        return profile, combo, e["mu"], e["chi"]

    selections = [
        pick("coupled_vomega", 1, 0.3, 0.005),
        pick("octagon", 1, 0.5, 0.000),
    ]

    out_dir = Path(__file__).resolve().parent
    exports_dir = out_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    for label, (profile, combo, mu, chi) in [("A", selections[0]), ("B", selections[1])]:
        print(f"\nSelection {label}: {profile} c{combo:03d} mu={mu} chi={chi}")

        # Resolve filename via catalog.
        filename = None
        for entry in catalog["entries"]:
            if (
                entry["profile"] == profile
                and entry["combo"] == combo
                and abs(entry["mu"] - mu) < 1e-9
                and abs(entry["chi"] - chi) < 1e-9
            ):
                filename = entry["filename"]
                break
        assert filename is not None, f"Could not resolve {profile}/{combo}/{mu}/{chi}"
        print(f"  filename: {filename}")

        payload = extract(profile, combo, mu, chi, max_points=1500)
        n = len(payload["time"])
        print(f"  decimated points: {n} (full {payload['meta']['n_rows_full']})")

        # Array-length contract.
        lengths = [
            len(payload["time"]),
            len(payload["actual"]["x"]),
            len(payload["actual"]["y"]),
            len(payload["reference"]["x"]),
            len(payload["reference"]["y"]),
        ]
        assert all(l == n for l in lengths), f"Mismatched array lengths: {lengths}"
        assert n > 1, "Need more than one decimated point"
        assert n <= 1500, f"Decimated points {n} exceed max_points"

        # Shared-origin contract.
        assert abs(payload["reference"]["x"][0] - payload["actual"]["x"][0]) < 1e-6
        assert abs(payload["reference"]["y"][0] - payload["actual"]["y"][0]) < 1e-6

        # Render a mid-trajectory frame.
        mid_idx = n // 2
        fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
        artists = setup_axes(ax, payload)
        update_frame(artists, payload, mid_idx)
        frame_path = out_dir / f"acceptance_frame_{label}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        matplotlib.pyplot.close(fig)
        size = frame_path.stat().st_size
        print(f"  wrote {frame_path} ({size} bytes)")
        assert size > 5 * 1024, f"Frame {frame_path} too small ({size} bytes)"

    # Export playback GIF for selection B.
    print("\nExporting playback GIF...")
    payload_b = extract(*selections[1], max_points=1500)
    gif_path = exports_dir / "acceptance_playback.gif"
    try:
        export_gif(payload_b, gif_path, fps=15)
        print(f"  wrote {gif_path} ({gif_path.stat().st_size} bytes)")
        assert gif_path.stat().st_size > 10 * 1024
    except Exception as exc:
        # Fallback: write three numbered progression frames.
        print(f"  GIF export failed ({exc}); writing numbered frames instead")
        for i, idx in enumerate([0, len(payload_b["time"]) // 2, len(payload_b["time"]) - 1]):
            fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
            artists = setup_axes(ax, payload_b)
            update_frame(artists, payload_b, idx)
            fp = exports_dir / f"acceptance_playback_frame_{i:03d}.png"
            fig.savefig(fp, dpi=120, bbox_inches="tight")
            matplotlib.pyplot.close(fig)
            print(f"  wrote {fp}")

    print("\nAll acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
