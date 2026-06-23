#!/usr/bin/env python
# =============================================================================
# blend_reports.py — merge the three per-screen reports into ONE combined CSV.
#
#   chatter_report.csv  +  sampling_sensitivity.csv  +  tracking_report.csv
#   -> diagnostics_combined.csv  (one row per trajectory, all metrics + a
#      single combined training recommendation)
#
# The three modules each stream/resume independently (their own CSV is the
# resume marker); this is the post-step that fuses their outputs. Run it after
# (re)running any of the three.
#
#   python blend_reports.py            # uses the default CSV names in cwd
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

META = ['profile', 'combo_idx', 'mu', 'friction_case', 'friction_model', 'chi']


def _combined_reco(r) -> str:
    """Single training recommendation fusing all three screens."""
    if r.get('chatter_verdict') == 'hash':
        return 'reject_hash'            # contaminated forces
    burst = str(r.get('burst_flag')) == 'True'
    track = r.get('track_flag')
    # A localized M7 control burst only rejects when tracking ALSO degraded
    # (marginal/missed). A burst on a well-tracked run is by-design excitation —
    # spin_creep's scheduled high-yaw-rate pulse edges spike Msat but the platform
    # tracks fine — not a give-up burst. Every genuine give-up burst coincides with
    # a tracking miss, so requiring both costs no real rejection (verified on the
    # sweep: all coupled_vomega/octagon bursts are 'missed'; the only burst+tracked
    # population is spin_creep's 66 pulse trajectories, which we keep).
    if burst and track in ('marginal', 'missed'):
        return 'reject_burst'           # give-up burst confirmed by lost tracking
    if track == 'missed':
        return 'reject_missed'          # did not achieve the commanded trajectory
    flags = []
    if r.get('chatter_verdict') == 'chatter':
        flags.append('chatter')
    if track == 'marginal':
        flags.append('marg_track')
    if burst:
        flags.append('burst')           # burst on a tracked run -> kept, labelled for provenance
    return 'keep_flagged:' + '+'.join(flags) if flags else 'keep'


def blend(chatter='chatter_report.csv', sampling='sampling_sensitivity.csv',
          tracking='tracking_report.csv', out='diagnostics_combined.csv') -> pd.DataFrame:
    ch = pd.read_csv(chatter).rename(columns={'verdict': 'chatter_verdict',
                                              'error': 'chatter_error'})
    # sampling/tracking: keep 'file' + their own columns; drop the duplicated meta
    sa = (pd.read_csv(sampling)
          .drop(columns=META + ['fs_native'], errors='ignore')
          .rename(columns={'verdict_native': 'sampling_native_verdict',
                           'error': 'sampling_error'}))
    tr = (pd.read_csv(tracking)
          .drop(columns=META, errors='ignore')
          .rename(columns={'error': 'tracking_error'}))
    df = ch.merge(sa, on='file', how='left').merge(tr, on='file', how='left')
    df['combined_reco'] = df.apply(_combined_reco, axis=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f'[blend] {len(df)} trajectories, {df.shape[1]} columns -> {out}')
    print('\n[combined recommendation]')
    print(df['combined_reco'].apply(lambda s: s.split(':')[0]).value_counts().to_string())
    print('\n[chatter_verdict x track_flag]')
    print(pd.crosstab(df['chatter_verdict'], df['track_flag']))
    keep = df[~df['combined_reco'].str.startswith('reject')]
    print(f"\n[whitelist] keep {len(keep)} / {len(df)}  (reject "
          f"{sum(df['combined_reco']=='reject_hash')} hash + "
          f"{sum(df['combined_reco']=='reject_burst')} burst + "
          f"{sum(df['combined_reco']=='reject_missed')} missed)")
    return df


def _cli():
    ap = argparse.ArgumentParser(description='Blend the three diagnostic reports into one CSV.')
    ap.add_argument('--chatter', default='chatter_report.csv')
    ap.add_argument('--sampling', default='sampling_sensitivity.csv')
    ap.add_argument('--tracking', default='tracking_report.csv')
    ap.add_argument('--out', default='diagnostics_combined.csv')
    a = ap.parse_args()
    blend(a.chatter, a.sampling, a.tracking, a.out)


if __name__ == '__main__':
    _cli()
