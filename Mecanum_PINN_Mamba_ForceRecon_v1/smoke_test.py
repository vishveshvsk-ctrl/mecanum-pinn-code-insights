"""Shape/wiring smoke test on random tensors (no data, CPU-OK). Run on the WSL
GPU box where torch is installed:  python smoke_test.py

Validates: forward force reconstruction shape, Heun NE residual + backward,
inverse Delta-window reconstruction shape, and the (mu_hat, chi_hat) readout.
"""
import torch

from mecanum_pinn import data
from mecanum_pinn.config import apply_dummy_overrides, build_config
from mecanum_pinn.losses import forward_losses, inverse_losses
from mecanum_pinn.models import MecanumPINN, mu_readout_residual
from mecanum_pinn.physics import RobotParams


def main():
    cfg = build_config(vram_gb=6, dummy=True)
    cfg['device'] = torch.device('cpu')
    apply_dummy_overrides(cfg)
    data.init_torch_globals(cfg['device'])
    rp = RobotParams().finalize(p1_wheels=0.11, device=cfg['device'])
    model = MecanumPINN(cfg, rp)

    B, L = 8, cfg['seq_len']
    S = torch.randn(B, L, 11); U = torch.randn(B, L, 4); T = torch.zeros(B, L, 1)
    S_next = torch.randn(B, L, 11); F_sim = torch.randn(B, L, 8)
    mu = torch.full((B,), 0.5); chi = torch.full((B,), 0.005)
    batch = (S, U, T, S_next, F_sim, mu, chi)

    # Forward
    F_phys, shapes = model.forward_model(S, U, mu, chi)
    assert F_phys.shape == (B, L, 8), F_phys.shape
    fl = forward_losses(F_phys, batch, rp, cfg)
    (fl['grnd'] + fl['phys']).backward()
    print("forward:", {k: round(float(v), 6) for k, v in fl.items()})

    # Inverse losses + TEST-TIME mu readout (residual; never trained)
    F_fwd, shapes = model.forward_model(S, U, mu, chi)
    F_inv = model.inverse_model(S, U)
    off = max(2, cfg['inv_window']) - 1
    assert F_inv.shape == (B, L - off, 8), F_inv.shape
    il = inverse_losses(F_inv, F_fwd.detach(), batch, rp, cfg)
    (il['grnd'] + il['cons'] + il['phys']).backward()
    print("inverse:", {k: round(float(v), 6) for k, v in il.items()})

    shapes_a = {k: v[:, off:] for k, v in shapes.items()}
    mu_inv, mu_conf = mu_readout_residual(F_inv, shapes_a, chi, rp.N_per_roller)
    mu_fwd, _ = mu_readout_residual(F_fwd[:, off:], shapes_a, chi, rp.N_per_roller)
    print("mu_hat(inv)[:3]:", mu_inv[:3].tolist())
    print("mu_hat(fwd)[:3]:", mu_fwd[:3].tolist())
    assert torch.isfinite(fl['phys']) and bool(torch.isfinite(mu_inv).all())
    print("\nSMOKE OK")


if __name__ == '__main__':
    main()
