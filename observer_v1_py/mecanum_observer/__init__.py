"""mecanum_observer — Approach 2: supervised causal state observer.

Reconstructs the unobservable per-wheel contact states (gamma roller-rate, LuGre
bristle zx/zy/zs) from measurable signals only, trained against Arrow-column
labels. omega_z is derived analytically from the gamma head. SSM (Mamba-lite
selective scan) vs GRU baseline; the per-state reconstruction error is the
observability signature.
"""
# Intentionally minimal: importing the package pulls in NOTHING heavy (no torch,
# no pandas/pyarrow). Import submodules explicitly as needed:
#   from mecanum_observer import config, data, features, models, training, ...
from . import config  # noqa: F401  (numpy-only)

__all__ = ["config"]
