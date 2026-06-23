"""Evaluation: test-set loss, and rolling (mu_hat, chi_hat) identification."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .losses import forward_losses, inverse_losses
from .models import mu_readout_residual


def _to_device(batch, device):
    return tuple(b.to(device) if torch.is_tensor(b) else b for b in batch)


def evaluate_on_test(model, loader, rp, config: Dict[str, Any],
                     stage: str, desc: str = '') -> Dict[str, float]:
    model.eval()
    device = config['device']
    sums: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            S, U, _T, S_next, F_sim, mu, chi = batch
            if stage == 'forward':
                F_phys, _shapes = model.forward_model(S, U, mu, chi)
                comp = forward_losses(F_phys, batch, rp, config)
            else:
                F_fwd, _shapes = model.forward_model(S, U, mu, chi)
                F_inv = model.inverse_model(S, U)
                comp = inverse_losses(F_inv, F_fwd, batch, rp, config)
            for k, v in comp.items():
                if torch.is_tensor(v) and v.dim() == 0:
                    sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    out = {k: v / max(1, n) for k, v in sums.items()}
    print(f"[eval/{desc or stage}] " + ' '.join(f"{k}={v:.3e}" for k, v in out.items()))
    return out


def estimate_mu(model, traj: Dict[str, Any], rp, config: Dict[str, Any]) -> float:
    """Rolling mu_hat over one trajectory via the test-time residual readout from F_inv,
    confidence-weighted by slip energy."""
    from .data import MecanumTrajectoryDataset, make_loader
    model.eval()
    device = config['device']
    ds = MecanumTrajectoryDataset([traj], seq_len=config['seq_len'],
                                  stride=config.get('stride', 1))
    if len(ds) == 0:
        return float('nan')
    loader = make_loader(ds, {**config, 'batch_size': min(256, len(ds)),
                              'num_workers': 0}, shuffle=False, drop_last=False)
    off = max(2, int(config['inv_window'])) - 1
    num = den = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            S, U, _T, _S_next, _F_sim, mu, chi = batch
            _F, shapes = model.forward_model(S, U, mu, chi)
            F_inv = model.inverse_model(S, U)
            shapes_a = {k: v[:, off:] for k, v in shapes.items()}
            mu_hat, mu_conf = mu_readout_residual(F_inv, shapes_a, chi, rp.N_per_roller)
            num += float((mu_conf * mu_hat).sum()); den += float(mu_conf.sum())
    return num / max(1e-9, den)


def evaluate_mu_id(model, loader, rp, config: Dict[str, Any], desc: str = ''
                   ) -> Dict[str, float]:
    """TEST-TIME mu identification over a loader (val/test). Slip-energy-weighted mu_hat
    from BOTH F_inv (identification) and F_fwd (self-consistency: should recover the
    conditioning mu). Reports MAE vs true mu and the F_inv-vs-F_fwd divergence (change signal).
    """
    model.eval()
    device = config['device']
    off = max(2, int(config['inv_window'])) - 1
    s_inv = s_fwd = s_div = s_w = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            S, U, _T, _S_next, _F_sim, mu, chi = batch
            F_fwd, shapes = model.forward_model(S, U, mu, chi)
            F_inv = model.inverse_model(S, U)
            shapes_a = {k: v[:, off:] for k, v in shapes.items()}
            mu_inv, c_inv = mu_readout_residual(F_inv, shapes_a, chi, rp.N_per_roller)
            mu_fwd, _cf   = mu_readout_residual(F_fwd[:, off:], shapes_a, chi, rp.N_per_roller)
            w = c_inv
            s_inv += float((w * (mu_inv - mu).abs()).sum())
            s_fwd += float((w * (mu_fwd - mu).abs()).sum())
            s_div += float((w * (mu_inv - mu_fwd).abs()).sum())
            s_w   += float(w.sum())
    out = {'mu_mae_inv': s_inv / max(1e-9, s_w),
           'mu_mae_fwd': s_fwd / max(1e-9, s_w),
           'mu_inv_fwd_div': s_div / max(1e-9, s_w)}
    print(f"[mu-id/{desc}] MAE(inv)={out['mu_mae_inv']:.3f} "
          f"MAE(fwd)={out['mu_mae_fwd']:.3f} inv-fwd div={out['mu_inv_fwd_div']:.3f}")
    return out
