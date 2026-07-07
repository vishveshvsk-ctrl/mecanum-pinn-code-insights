# =============================================================================
# hybrid_ctrl/fuzzy.jl — type-1 triangular fuzzy supervisor
# =============================================================================
module FuzzyMod

using StaticArrays

export FuzzySupervisor, fuzzy_update!

"""
    FuzzySupervisor

Type-1 triangular membership functions on (e_p, e_v, ė_p).  Outputs normalized
blend weights over the enabled controllers.  Reduces to paper β(k) when only
MPC+PID are enabled.
"""
Base.@kwdef mutable struct FuzzySupervisor
    ep_max::Float64   = 0.5
    ev_max::Float64   = 0.3
    dep_max::Float64  = 0.2
    rate_hz::Float64  = 50.0
end

"Triangular MF: center c, half-width h, value at x."
function _tri(x, c, h)
    return max(0.0, 1.0 - abs(x - c) / h)
end

"Singleton rule outputs (preference for ASMC/MPC/PID)."
const _RULE_OUT = Dict(
    # (ep_size, ev_size, dep_sign) => (w_ASMC, w_MPC, w_PID)
    (:S, :S, :N) => (0.2, 0.5, 0.3),
    (:S, :S, :Z) => (0.1, 0.6, 0.3),
    (:S, :S, :P) => (0.2, 0.5, 0.3),
    (:S, :M, :N) => (0.3, 0.5, 0.2),
    (:S, :M, :Z) => (0.2, 0.6, 0.2),
    (:S, :M, :P) => (0.3, 0.5, 0.2),
    (:S, :L, :N) => (0.5, 0.4, 0.1),
    (:S, :L, :Z) => (0.4, 0.5, 0.1),
    (:S, :L, :P) => (0.5, 0.4, 0.1),
    (:M, :S, :N) => (0.3, 0.4, 0.3),
    (:M, :S, :Z) => (0.2, 0.5, 0.3),
    (:M, :S, :P) => (0.3, 0.4, 0.3),
    (:M, :M, :N) => (0.4, 0.4, 0.2),
    (:M, :M, :Z) => (0.3, 0.5, 0.2),
    (:M, :M, :P) => (0.4, 0.4, 0.2),
    (:M, :L, :N) => (0.6, 0.3, 0.1),
    (:M, :L, :Z) => (0.5, 0.4, 0.1),
    (:M, :L, :P) => (0.6, 0.3, 0.1),
    (:L, :S, :N) => (0.6, 0.2, 0.2),
    (:L, :S, :Z) => (0.5, 0.3, 0.2),
    (:L, :S, :P) => (0.6, 0.2, 0.2),
    (:L, :M, :N) => (0.7, 0.2, 0.1),
    (:L, :M, :Z) => (0.6, 0.3, 0.1),
    (:L, :M, :P) => (0.7, 0.2, 0.1),
    (:L, :L, :N) => (0.8, 0.1, 0.1),
    (:L, :L, :Z) => (0.7, 0.2, 0.1),
    (:L, :L, :P) => (0.8, 0.1, 0.1),
)

_size_label(x, maxval) = begin
    m = abs(x)
    if m < 0.33*maxval; return :S
    elseif m < 0.66*maxval; return :M
    else; return :L
    end
end

_dep_label(dep) = dep < -0.1 ? :N : (dep > 0.1 ? :P : :Z)

function fuzzy_update!(bus, xhat, ref, cfg, fz::FuzzySupervisor)
    t = bus.t_now[]
    if cfg.tracking == :velocity
        ep = hypot(xhat[1] - ref.Vx(t), xhat[2] - ref.Vy(t))
        ev = hypot(xhat[1] - ref.Vx(t), xhat[2] - ref.Vy(t))
        dep = xhat[3] - ref.Wz(t)
    else
        ep = hypot(xhat[5] - ref.xo(t), xhat[6] - ref.yo(t))
        ev = hypot(xhat[1] - ref.Vxo(t), xhat[2] - ref.Vyo(t))
        dep = xhat[3] - ref.om(t)
    end

    if !cfg.fuzzy
        bus.weights = SVector(cfg.fixed_weights[1], cfg.fixed_weights[2], cfg.fixed_weights[3])
        return bus.weights
    end

    ep_l = _size_label(ep, fz.ep_max)
    ev_l = _size_label(ev, fz.ev_max)
    dep_l = _dep_label(dep)
    key = (ep_l, ev_l, dep_l)

    out = get(_RULE_OUT, key, (0.34, 0.33, 0.33))
    w_raw = SVector(out[1], out[2], out[3])

    # Zero out disabled controllers
    en = SVector(cfg.use_asmc ? 1.0 : 0.0,
                 cfg.use_mpc  ? 1.0 : 0.0,
                 cfg.use_pid  ? 1.0 : 0.0)
    w_masked = w_raw .* en
    s = sum(w_masked)
    if s > 1e-6
        bus.weights = w_masked / s
    else
        bus.weights = en / max(sum(en), 1.0)
    end
    return bus.weights
end

end # module
