"""Mecanum-PINN training package (modular v3).

Top-level usage from a thin script:

    from mecanum_pinn import run_main

    run_main(
        config_kwargs=dict(vram_gb=6, motion_cases=['circle','infinity']),
        prefix='exp1', suffix='seed42',
    )

For interactive / notebook use, the most-needed names are re-exported here:
configuration, the model, the new world-frame trajectory evaluator, and
the figure helpers. Anything else lives in its module file -- for example,
import mecanum_pinn.training to get at run_lbfgs_refine or _epoch_loop.
"""

__version__ = "12.5.0"

# To avoid import order issues from pyarrow
import pyarrow.feather

# --- top-level entry point ----------------------------------
from .stages import run_main

# --- config + run-tag ---------------------------------------
from .config import (
    apply_dummy_overrides,
    build_config,
    build_run_tag,
)

# --- manifest -----------------------------------------------
from .manifest import (
    load_manifest_at,
    load_training_manifest,
    save_training_manifest,
)

# --- physics + models ---------------------------------------
from .physics import (
    Geometry,
    RobotParams,
    make_geometry,
)
from .models import (
    MecanumForwardModel,
    MecanumInverseModel,
    MecanumPINN,
    build_empty_pinn,
    maybe_compile_pinn,
)

# --- data + loaders -----------------------------------------
from .data import (
    MecanumTrajectoryDataset,
    build_loaders,
    build_loaders_from_lists,
    build_loaders_with_split,
    filter_trajectories_by_name,
    init_torch_globals,
    load_all_arrow_trajectories,
    load_ood_test_trajectories,
    parse_whitelist,
    stratified_split,
)

# --- training -----------------------------------------------
from .training import (
    train_forward,
    train_inverse,
    train_inverse_ablation,
)

# --- evaluation + plotting ----------------------------------
from .evaluation import (
    estimate_and_plot_mu_chi,
    estimate_mu_chi,
    evaluate_on_test,
    evaluate_ood,
    plot_test_trajectory_predictions,
)
from .plotting import (
    configure_figure_saving,
    plot_history,
    plot_id_vs_ood_comparison,
    plot_train_val_test_comparison,
    save_figure,
)

# --- world-frame trajectory evaluation (new) ----------------
from .trajectory_eval import (
    evaluate_and_plot_trajectory_window,
    evaluate_trajectory_window,
    plot_trajectory_window,
)

__all__ = [
    "__version__",
    # entry point
    "run_main",
    # config
    "apply_dummy_overrides", "build_config", "build_run_tag",
    # manifest
    "load_manifest_at", "load_training_manifest", "save_training_manifest",
    # physics + models
    "Geometry", "RobotParams", "make_geometry",
    "MecanumForwardModel", "MecanumInverseModel", "MecanumPINN",
    "build_empty_pinn", "maybe_compile_pinn",
    # data
    "MecanumTrajectoryDataset", "build_loaders",
    "build_loaders_from_lists", "build_loaders_with_split",
    "filter_trajectories_by_name", "init_torch_globals",
    "load_all_arrow_trajectories", "load_ood_test_trajectories",
    "parse_whitelist", "stratified_split",
    # training
    "train_forward", "train_inverse", "train_inverse_ablation",
    # evaluation + plotting
    "estimate_and_plot_mu_chi", "estimate_mu_chi",
    "evaluate_on_test", "evaluate_ood",
    "plot_test_trajectory_predictions",
    "configure_figure_saving", "plot_history",
    "plot_id_vs_ood_comparison", "plot_train_val_test_comparison",
    "save_figure",
    # trajectory eval
    "evaluate_and_plot_trajectory_window",
    "evaluate_trajectory_window",
    "plot_trajectory_window",
]
