# =============================================================================
# hybrid_ctrl_v2/controller_tuning/mpc_design.jl — MPCDesignMod (Stage 3)
# =============================================================================
# Analytic MPC weight construction: Bryson-rule seeding + DARE terminal
# weight + the reduced Stage-3 search space. Depends on
# `Main.ControllerMod._mpc_body_matrices`/`_mpc_pose_A` (unexported but
# qualification-accessible -- `hybrid_ctrl/controllers.jl` is never edited).
#
# `MatrixEquations` is now a direct Project.toml dependency (added via
# `Pkg.add`, per-user direction). `terminal_weight` still degrades to a zero
# matrix + `@warn` if the DARE solve itself fails (e.g. an unstabilizable
# linearization) -- matching the brief's "must degrade gracefully" contract,
# just triggered by solver failure rather than package absence now.
# =============================================================================
module MPCDesignMod

using StaticArrays
using LinearAlgebra
using MatrixEquations

export bryson_weights, terminal_weight, normalized_mpc_space, expand_ratio_space, self_check

"""
    bryson_weights(tol::NamedTuple, v_max::Float64) -> (Q_pose::SVector{6}, R::SVector{4})

Bryson's rule: Q_ii = 1/(max acceptable deviation_i)^2, R_ii = 1/V_max^2.
Pose-state order (xo,yo,psi,Vx,Vy,Wz) matches `ControllerMod.mpc_wrench!`'s
`x0` construction. Position/heading use `tol.pos_max`/`tol.head_max`
directly -- they ARE Bryson max-deviations by construction. The velocity
sub-block has no equivalent in `TOL` (`vel_rms`/`yawrate_rms` are RMS
*targets* ~1e-3, not deviation ceilings -- using them directly would make
q_vel ~1e6, three orders above the position/heading entries and structurally
different in kind); 10x the RMS target is used instead as the velocity
allowable-deviation proxy, keeping the same tolerance-derived spirit without
a degenerate entry.
"""
function bryson_weights(tol::NamedTuple, v_max::Float64)
    q_pos  = 1.0 / tol.pos_max^2
    q_head = 1.0 / tol.head_max^2
    q_vel  = 1.0 / (10 * tol.vel_rms)^2
    q_yawr = 1.0 / (10 * tol.yawrate_rms)^2
    Q_pose = SVector(q_pos, q_pos, q_head, q_vel, q_vel, q_yawr)
    R = SVector{4}(fill(1.0 / v_max^2, 4))
    return Q_pose, R
end

"""
    terminal_weight(params, motor, dt, Q, R) -> Matrix{Float64}

Discrete-algebraic-Riccati terminal weight for the 6-state pose model
(`ControllerMod._mpc_pose_A`, linearized at zero body velocity/heading -- a
fixed nominal operating point since this is a DESIGN constant, not
re-linearized per tick). Returns a 6x6 zero matrix + `@warn` if the DARE solve
fails (e.g. an unstabilizable linearization at the chosen operating point).
"""
function terminal_weight(params, motor, dt::Float64, Q, R)
    try
        _, Bvel, _ = Main.ControllerMod._mpc_body_matrices(params, motor, dt)
        A = Matrix(Main.ControllerMod._mpc_pose_A(params, dt, SVector(0.0, 0.0, 0.0), 0.0))
        B = zeros(6, 4);  B[4:6, :] .= Matrix(Bvel)
        Qm = Matrix(Diagonal(collect(Q)))
        Rm = Matrix(Diagonal(collect(R)))
        P, _, _ = MatrixEquations.ared(A, B, Rm, Qm)
        return Matrix(P)
    catch e
        @warn "MPCDesignMod.terminal_weight: DARE solve failed -- Stage 3 runs with NO terminal cost (P_terminal = 0)" exception = e
        return zeros(6, 6)
    end
end

"""
    normalized_mpc_space() -> Vector{Tuple}

Stage-3 search space: RATIO parameters only, relative to the Bryson-seeded
`(Q_pose, R)` from `bryson_weights` (design values, not search variables --
see brief). `R_scale` is PINNED to 1.0: scaling `(Q,R,S)` by any c>0 scales
both the QP's H and f by c and leaves the argmin/constraints unchanged (a
flat direction that degenerates dxNES's covariance adaptation -- verified by
the `self_check` scaling-invariance test below). Velocity-mode `Q` is
EXCLUDED: every tier trajectory runs `mode=:pose` this iteration, so
`mpc.Q` is never read (verified by `self_check`'s dead-Q test).

Two ratio dimensions remain non-flat and therefore searchable:
  - `S_scale`      voltage-RATE weight relative to the pinned R=1 -- trades
                   control smoothness against tracking, NOT flat (unlike a
                   uniform (Q,R,S) rescale).
  - `q_vel_ratio`  relative weight on the Q_pose velocity sub-block vs the
                   Bryson position/heading sub-block -- shifts where the
                   finite horizon spends tracking authority (velocity vs
                   pose channels), also not a uniform-rescale direction.
`Np_pose` is swept as a separate outer grid {10,15,20,30} (brief §8 step 17),
not part of this continuous space.
"""
function normalized_mpc_space()
    return [
        ("S_scale",     1, :log, 1e-3, 1.0),
        ("q_vel_ratio", 1, :log, 0.1,  10.0),
    ]
end

"""
    expand_ratio_space(kw, base_Q_pose, base_R, base_S) -> NamedTuple

`StageObjectiveMod.make_stage_objective`'s `expand` hook for Stage 3: turns
the two RATIO parameters `normalized_mpc_space()` searches into the actual
`Q_pose`/`R`/`S` `build_controller_v2` needs, relative to the Bryson-seeded
`(base_Q_pose, base_R)` and a nominal `base_S`. `R` itself is NOT rescaled by
`S_scale` (stays pinned at `base_R`; only `S` moves relative to it, per
`normalized_mpc_space`'s docstring). Any OTHER keys already in `kw` (e.g. a
frozen `Np_pose`/`P_terminal`) pass through unchanged.
"""
function expand_ratio_space(kw::NamedTuple, base_Q_pose, base_R, base_S)
    q_vel_ratio = get(kw, :q_vel_ratio, 1.0)
    S_scale     = get(kw, :S_scale, 1.0)
    Q_pose = SVector(base_Q_pose[1], base_Q_pose[2], base_Q_pose[3],
                     base_Q_pose[4]*q_vel_ratio, base_Q_pose[5]*q_vel_ratio, base_Q_pose[6]*q_vel_ratio)
    S = SVector{4}(collect(base_S) .* S_scale)
    return merge(kw, (Q_pose=Q_pose, R=base_R, S=S))
end

"""
    self_check(params, motor, dt) -> NamedTuple

Stage-3 unit tests (brief §10 Success Criteria):
  1. scaling `(Q_pose, R, S)` by any c>0 leaves the applied first-step
     voltage `u0` unchanged to solver tolerance (confirms the flat direction
     `R_scale` is pinned against).
  2. with `mode=:pose`, mutating `mpc.Q` (velocity-mode weight) changes no
     output (confirms those 3 params are dead and correctly excluded).
Requires `ControllerV2Mod.MPCControllerV2` + `Main.ControllerMod.mpc_wrench!`
to already be loaded (i.e. run after `include("controllers_v2.jl")`).
"""
function self_check(bus, xhat, ref, params, motor)
    mpc1 = Main.ControllerV2Mod.MPCControllerV2(Np_pose=8)
    W1 = Main.ControllerMod.mpc_wrench!(deepcopy(bus), xhat, ref, params, motor, mpc1; mode=:pose)

    c = 7.0
    mpc2 = Main.ControllerV2Mod.MPCControllerV2(Np_pose=8, Q_pose=c .* mpc1.Q_pose,
                                                R=c .* mpc1.R, S=c .* mpc1.S)
    W2 = Main.ControllerMod.mpc_wrench!(deepcopy(bus), xhat, ref, params, motor, mpc2; mode=:pose)
    scale_invariant = isapprox(collect(W1), collect(W2); atol=1e-4, rtol=1e-4)

    mpc3 = Main.ControllerV2Mod.MPCControllerV2(Np_pose=8, Q=SVector(1e4, 1e4, 1e4))
    W3 = Main.ControllerMod.mpc_wrench!(deepcopy(bus), xhat, ref, params, motor, mpc3; mode=:pose)
    q_dead = isapprox(collect(W1), collect(W3); atol=1e-9, rtol=1e-9)

    return (scale_invariant=scale_invariant, q_dead=q_dead, W1=W1, W2=W2, W3=W3)
end

end # module
