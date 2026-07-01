"""Top-level entry point for the Mamba ForceRecon PINN.

Usage (from code_insights/):
    python Mecanum_PINN_Mamba_ForceRecon_v1/train.py both      # forward then inverse
    python Mecanum_PINN_Mamba_ForceRecon_v1/train.py forward
    python Mecanum_PINN_Mamba_ForceRecon_v1/train.py inverse --ckpt Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/<tag>/forward_lbfgs.pth
    python Mecanum_PINN_Mamba_ForceRecon_v1/train.py figures --ckpt Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/<tag>/inverse_lbfgs.pth

CLI overrides (consumed by the shared parallel launcher; see launch_parallel.py):
    --vram {6,12,24} --regime <toml> --test-chi <f> --batch-size/--per-run-batch <n>
    --cache-dir <dir> --run-tag <tag> --no-lbfgs --warm-cache-only --set KEY=VALUE

GPU settings (compile, matmul precision, cudnn.benchmark, VRAM-tier batch sizes)
are retained from train_GPU_PINN_v14. `vram_gb` defaults to 24 (Quadro RTX 6000)
and is overridable with --vram. The decimated-trajectory cache is ON by default
inside the package (`Mecanum_PINN_Mamba_ForceRecon_v1/cache_decim`); override the
location with --cache-dir, disable with `--cache-dir ''`. Point `whitelist_path`
at the list exported from diagnostics_combined.csv.

Checkpoints, figures, metrics.json and manifest.json are written under
`Mecanum_PINN_Mamba_ForceRecon_v1/runs/` so they stay inside this package rather
than spilling into the parent code_insights/ directory.
"""
from mecanum_pinn.stages import run_main

if __name__ == '__main__':
    run_main(
        config_kwargs=dict(
            vram_gb=24,
            # cache_dir, ckpt_dir and figure_dir default inside this package;
            # data_dir / whitelist_path default to the lugre_adamov sweep +
            # pinn_training_whitelist.txt; override here or via CLI if needed.
        ),
        prefix='mamba',
        p1_wheels=0.11,        # drivetrain viscous (friction_case 1)
    )
