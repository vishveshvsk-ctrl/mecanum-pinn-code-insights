#!/usr/bin/env python3
"""
plot_controller_eskf_traces.py

Read Arrow traces produced by save_controller_eskf_traces.jl and generate
comparison figures for ASMC vs PID under clean/noisy feedback, one figure per
trajectory. Feedback can come from the frozen ESKF or from an oracle (true
state + optional noise).

Hard-coded to use the claude-venv Python in this repo's Windows tree.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_arrow(path: Path) -> pd.DataFrame:
    return pa.ipc.open_file(path).read_pandas()


def find_traces(trace_dir: Path):
    """Return dict: trajectory -> { (controller, noise_label) -> df }."""
    traces = {}
    for p in sorted(trace_dir.glob('*.arrow')):
        # fname: asmc_octagon_clean_seed42.arrow  (traj may contain underscores)
        stem = p.stem
        # strip seed suffix if present
        if stem.endswith('_seed42'):
            stem = stem[:-7]
        # noise is the last token, which must be clean or realistic
        parts = stem.split('_')
        if parts[-1] not in ('clean', 'realistic'):
            continue
        noise = parts[-1]
        ctrl = parts[0]
        traj = '_'.join(parts[1:-1])
        traces.setdefault(traj, {})[(ctrl, noise)] = p
    return traces


def cm(x):
    """Convert metres to centimetres."""
    return x * 100.0


def mrad(x):
    """Convert radians to milliradians."""
    return x * 1000.0


def composite_tracking(pos_err_cm, head_err_mrad):
    """Match tune_controller.jl TOL-normalized tracking score.

    TOL: pos_final=1 cm, pos_max=10 cm, head_final=10 mrad, head_max=100 mrad.
    """
    final_pos = pos_err_cm.iloc[-1]
    max_pos = pos_err_cm.max()
    final_head = head_err_mrad.iloc[-1]
    max_head = head_err_mrad.max()
    return (final_pos / 1.0 + max_pos / 10.0 +
            final_head / 10.0 + max_head / 100.0) / 4.0


# Title templates per feedback mode.
# Each maps noise_label -> (ASMC title, PID title).
TRAJ_TITLES = {
    'eskf': {
        'clean':     ('ASMC via clean ESKF',     'PID via clean ESKF'),
        'realistic': ('ASMC via noisy ESKF',     'PID via noisy ESKF'),
    },
    'oracle': {
        'clean':     ('ASMC via clean oracle',   'PID via clean oracle'),
        'realistic': ('ASMC via noisy oracle',   'PID via noisy oracle'),
    },
}


def plot_trajectory_figure(traj, data, out_path: Path, feedback: str = 'eskf'):
    """2x2 XY trajectory figure: ASMC clean, PID clean, ASMC noisy, PID noisy."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    fig.suptitle(f'Controller comparison — {traj}', fontsize=13, fontweight='bold')

    titles = TRAJ_TITLES[feedback]
    order = [
        ('asmc', 'clean',     titles['clean'][0]),
        ('pid',  'clean',     titles['clean'][1]),
        ('asmc', 'realistic', titles['realistic'][0]),
        ('pid',  'realistic', titles['realistic'][1]),
    ]

    colors = {
        'ref':  '#000000',
        'true': '#1f77b4',
        'est':  '#ff7f0e',
    }

    est_label = 'ESKF estimate' if feedback == 'eskf' else 'oracle estimate'

    for ax, (ctrl, noise, title) in zip(axes.flat, order):
        key = (ctrl, noise)
        if key not in data:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'missing trace', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        df = data[key]
        # Reference (dashed, opaque)
        ax.plot(cm(df['x_ref']), cm(df['y_ref']), '--', color=colors['ref'],
                lw=1.5, alpha=0.9, label='reference')
        # True trajectory (solid, fairly opaque)
        ax.plot(cm(df['x_true']), cm(df['y_true']), '-', color=colors['true'],
                lw=1.2, alpha=0.85, label='true (plant)')
        # Estimated/feedback trajectory (dotted, most translucent)
        ax.plot(cm(df['x_est']), cm(df['y_est']), ':', color=colors['est'],
                lw=1.0, alpha=0.35, label=est_label)

        # Compute metrics that match the summary table
        max_pos_err = cm(df['pos_err_true'].max())
        track_score = composite_tracking(cm(df['pos_err_true']),
                                         mrad(df['head_err_true']))
        ax.set_title(f'{title}\ntracking = {track_score:.3f} | max pos err = {max_pos_err:.2f} cm')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  saved {out_path.name}')


# Error-figure title templates per feedback mode.
ERROR_NOISE_TITLES = {
    'eskf':   {'clean': 'clean ESKF feedback',   'realistic': 'noisy ESKF feedback'},
    'oracle': {'clean': 'clean oracle feedback', 'realistic': 'noisy oracle feedback'},
}


def plot_error_figure(traj, data, out_path: Path, feedback: str = 'eskf'):
    """2x2 error figure: position error (top) and heading error (bottom)
       for clean (left) and realistic (right), ASMC vs PID overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')
    fig.suptitle(f'Tracking errors — {traj}', fontsize=13, fontweight='bold')

    colors = {'asmc': '#1f77b4', 'pid': '#d62728'}
    titles = ERROR_NOISE_TITLES[feedback]
    est_suffix = 'ESKF' if feedback == 'eskf' else 'oracle'

    noise_cols = [('clean', titles['clean']), ('realistic', titles['realistic'])]
    controllers = [('asmc', 'ASMC'), ('pid', 'PID')]

    for j, (noise, noise_title) in enumerate(noise_cols):
        ax_pos = axes[0, j]
        ax_head = axes[1, j]

        for ctrl, ctrl_label in controllers:
            key = (ctrl, noise)
            if key not in data:
                continue
            df = data[key]
            t = df['t']
            ax_pos.plot(t, cm(df['pos_err_true']), '-', color=colors[ctrl],
                        lw=1.0, label=f'{ctrl_label} true')
            # estimate error can differ from true error under noise
            ax_pos.plot(t, cm(df['pos_err_est']), ':', color=colors[ctrl],
                        lw=1.0, label=f'{ctrl_label} {est_suffix}')

            ax_head.plot(t, mrad(df['head_err_true']), '-', color=colors[ctrl],
                         lw=1.0, label=f'{ctrl_label} true')
            ax_head.plot(t, mrad(df['head_err_est']), ':', color=colors[ctrl],
                         lw=1.0, label=f'{ctrl_label} {est_suffix}')

        ax_pos.set_title(noise_title)
        ax_pos.set_ylabel('position error (cm)')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc='best', fontsize=8)

        ax_head.set_xlabel('time (s)')
        ax_head.set_ylabel('heading error (mrad)')
        ax_head.grid(True, alpha=0.3)
        ax_head.legend(loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  saved {out_path.name}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot controller feedback simulation traces.')
    parser.add_argument('--trace-dir',
                        default='hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose_traces',
                        help='Directory containing .arrow trace files')
    parser.add_argument('--out-dir',
                        default='hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose_figures',
                        help='Directory for output PNGs')
    parser.add_argument('--feedback', choices=['eskf', 'oracle'], default='eskf',
                        help='Feedback source shown in titles (default: eskf)')
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = find_traces(trace_dir)
    if not traces:
        print(f'No .arrow traces found in {trace_dir}')
        return

    print(f'Found traces for trajectories: {", ".join(sorted(traces.keys()))}')
    print(f'Writing figures to {out_dir}  (feedback mode: {args.feedback})')

    for traj in sorted(traces.keys()):
        print(f'\n{traj}:')
        data = {k: read_arrow(p) for k, p in traces[traj].items()}

        traj_path = out_dir / f'controller_{args.feedback}_pose_{traj}_trajectory.png'
        plot_trajectory_figure(traj, data, traj_path, feedback=args.feedback)

        err_path = out_dir / f'controller_{args.feedback}_pose_{traj}_errors.png'
        plot_error_figure(traj, data, err_path, feedback=args.feedback)

    print('\nDone.')


if __name__ == '__main__':
    main()
