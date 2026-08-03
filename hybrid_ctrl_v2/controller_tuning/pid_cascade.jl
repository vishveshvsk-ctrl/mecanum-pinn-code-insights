# =============================================================================
# hybrid_ctrl_v2/controller_tuning/pid_cascade.jl — PIDCascadeMod (Stage 2)
# =============================================================================
# Leaf module (no `include`s; consumed by StageObjectiveMod via a decode hook
# -- see `regroup_joint` below). Provides the three cascade-phase parameter
# spaces (brief §8 step 12): :inner (velocity loop, outer frozen at
# feedforward-only) -> :outer (inner frozen at the :inner winner) -> :joint
# (narrow box polish around the sequential optimum, all 18 params).
#
# :inner/:outer reuse tune_controller.jl's PID_SPACE row format/bounds
# verbatim (grouped len=3 rows -> `decode` already produces the SVector{3}
# fields `build_controller` wants). :joint needs a DIFFERENT [lo,hi] PER AXIS
# (a narrow box around each axis's own sequential-optimum value, not a shared
# row bound) -- `decode`/`flat_bounds` in tune_controller.jl only support one
# shared (lo,hi) per grouped row, and are reused unmodified (preservation
# constraint), so :joint instead uses 18 per-axis scalar rows ("Kp1".."Kd_pos3")
# and `regroup_joint` folds `decode`'s scalar fields back into the SVector{3}
# fields `build_controller` expects -- the "decode hook" the brief's
# StageObjectiveMod component description refers to.
# =============================================================================
module PIDCascadeMod

using StaticArrays

export pid_phase_space, regroup_joint

# Joint-phase narrow box: [0.3x, 3x] around the sequential (:inner + :outer)
# optimum for every one of the 18 axes -- "narrow" relative to the original
# PID_SPACE's ~10-160x lo/hi spans, while still wide enough that a genuinely
# better joint-polish point one order of magnitude off the sequential guess
# stays reachable.
const _LO_MULT, _HI_MULT = 0.3, 3.0

_narrow_bounds(val::Real) = (max(val * _LO_MULT, 1e-6), val * _HI_MULT)

"""
    pid_phase_space(phase::Symbol, frozen::NamedTuple) -> (space, freeze)

`phase`:
    :inner -> Kp, Ki, Kd, I_max         (3+3+3+3 = 12 params); `frozen` unused
    :outer -> Kp_pos, Kd_pos           (3+3 = 6 params); needs
              frozen.(Kp,Ki,Kd,I_max) from the :inner winner
    :joint -> all 18, narrow per-axis box; needs
              frozen.(Kp,Ki,Kd,I_max,Kp_pos,Kd_pos) from the sequential winner

`space` uses tune_controller.jl's row format `(name, len, scale, lo, hi)` for
:inner/:outer (len=3, decodes straight to an SVector{3}); :joint uses 18
len=1 rows and REQUIRES `regroup_joint` to be applied to `decode`'s output.
"""
function pid_phase_space(phase::Symbol, frozen::NamedTuple)
    if phase == :inner
        space = [
            ("Kp",    3, :log, 5.0,  800.0),
            ("Ki",    3, :log, 0.1,  150.0),
            ("Kd",    3, :log, 0.05, 50.0),
            ("I_max", 3, :log, 5.0,  200.0),
        ]
        # Outer loop pinned at feedforward-only (zero position-error gains) so
        # the inner velocity loop is isolated exactly like the time-scale-
        # separated cascade design assumes -- brief §8 step 12.
        freeze = (Kp_pos=SVector(0.0, 0.0, 0.0), Kd_pos=SVector(0.0, 0.0, 0.0))
        return space, freeze

    elseif phase == :outer
        for k in (:Kp, :Ki, :Kd, :I_max)
            haskey(frozen, k) || error("PIDCascadeMod.pid_phase_space(:outer, ...): " *
                                       "frozen.$k required (from the :inner winner)")
        end
        space = [
            ("Kp_pos", 3, :log, 0.2, 5.0),
            ("Kd_pos", 3, :log, 0.1, 3.0),
        ]
        freeze = (Kp=frozen.Kp, Ki=frozen.Ki, Kd=frozen.Kd, I_max=frozen.I_max)
        return space, freeze

    elseif phase == :joint
        for k in (:Kp, :Ki, :Kd, :I_max, :Kp_pos, :Kd_pos)
            haskey(frozen, k) || error("PIDCascadeMod.pid_phase_space(:joint, ...): " *
                                       "frozen.$k required (from the sequential :inner+:outer winner)")
        end
        rows = Tuple[]
        for (base, vec) in (("Kp", frozen.Kp), ("Ki", frozen.Ki), ("Kd", frozen.Kd),
                            ("I_max", frozen.I_max), ("Kp_pos", frozen.Kp_pos), ("Kd_pos", frozen.Kd_pos))
            for i in 1:3
                lo, hi = _narrow_bounds(Float64(vec[i]))
                push!(rows, ("$(base)$(i)", 1, :log, lo, hi))
            end
        end
        return rows, (;)   # nothing frozen -- every axis is searched, just narrowly
    else
        error("PIDCascadeMod.pid_phase_space: unknown phase $phase (expected :inner|:outer|:joint)")
    end
end

"""
    regroup_joint(kw::NamedTuple) -> NamedTuple

Fold the `:joint` phase's 18 per-axis scalar fields (as produced by
`decode(theta, joint_space)`, e.g. `kw.Kp1, kw.Kp2, kw.Kp3`) back into the
SVector{3} fields (`Kp`, `Ki`, ...) that `build_controller` expects. A no-op
pass-through for any NamedTuple that already has grouped fields (so callers
can apply it unconditionally without branching on phase).
"""
function regroup_joint(kw::NamedTuple)
    haskey(kw, :Kp1) || return kw
    grp(base) = SVector(getfield(kw, Symbol(base, 1)), getfield(kw, Symbol(base, 2)), getfield(kw, Symbol(base, 3)))
    return (Kp=grp(:Kp), Ki=grp(:Ki), Kd=grp(:Kd), I_max=grp(:I_max),
            Kp_pos=grp(:Kp_pos), Kd_pos=grp(:Kd_pos))
end

end # module
