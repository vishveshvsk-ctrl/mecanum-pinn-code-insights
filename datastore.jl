# =============================================================================
# datastore.jl — DataStore module: everything between `sol` and disk.
#
# Owns: label extraction (compute_labels), long-form DataFrame assembly,
# Arrow(+zstd)/JLD2 writing, the output FILENAME SCHEME (single source of
# truth for both the writer and the driver's resume check), the streaming
# state logger, and the reload helper.
#
# Deliberately physics-free: the three plant/controller callables that
# compute_labels needs (the LuGre rate function, the controller, and the
# sawtooth smoother) are INJECTED as keyword arguments, because they are
# defined in the notebook (Main), not here. This also fixes a latent bug in
# the old notebook cell, which hardcoded `asmc_torques` in the label pass even
# when the run used `asmc_torques_vel` — now the caller passes the controller
# that actually produced the run.
#
# meta:: NamedTuple contract (used by assemble_dataframe / naming / writing):
#   (profile        :: String,   # e.g. "octagon"
#    combo_idx      :: Int,      # 1-based row in [profile.combos]
#    mu             :: Float64,
#    chi            :: Float64,
#    friction_case  :: Int,
#    friction_model :: Symbol,   # :lugre_adamov | :lugre_uncoupled
#    sweep_seed     :: Int)
#
# Deps (all already in Project.toml): DataFrames, Arrow, JLD2, StaticArrays,
# Printf. CSV output is intentionally GONE (Arrow is the consumed format).
# =============================================================================
module DataStore

using DataFrames
using Arrow
using JLD2
using StaticArrays
using Printf
using DiffEqCallbacks: PresetTimeCallback
using Base.Threads: @spawn

export compute_labels, assemble_dataframe, write_outputs,
       output_prefix, expected_output,
       build_streaming_logger, reload_run,
       ntuple_from_params, ntuple_from_asmc

# =============================================================================
# Filename scheme — THE single source of truth.
#
#   <profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
#   e.g.  octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
#
# The driver's resume check calls expected_output(); the writer derives its
# paths from the same output_prefix(). They cannot drift.
# NOTE for the Python side: data.py's parse_arrow_filename / _FNAME_RE must be
# updated to this scheme (profile + combo replace beta + amp).
# =============================================================================
function output_prefix(outdir::AbstractString, meta::NamedTuple)
    joinpath(outdir, @sprintf("%s_c%03d_mu_%g_case%d_%s",
                              meta.profile, meta.combo_idx, meta.mu,
                              meta.friction_case, String(meta.friction_model)))
end

"""Full .arrow path a finished job must produce — used for resume detection."""
expected_output(outdir::AbstractString, meta::NamedTuple) =
    @sprintf("%s_chi_%.3f.arrow", output_prefix(outdir, meta), meta.chi)

# =============================================================================
# Label extraction — friction forces, slips, controller torques, DOB estimate.
# Body copied from the notebook cell (39-D state); physics enters ONLY through
# the injected callables:
#   friction_fn   = lugre_dyn_rates      (lugre, coupling, f_c, N_i, chi,
#                                         wz, Vpx, Vpy, zx, zy, zs) -> 6-tuple
#   controller_fn = asmc_torques | asmc_torques_vel
#                                         (u, t, params, asmc, eso) -> 4-tuple
#   sawtooth_fn   = sawtooth_approx      (θ,) -> θ̃
# `coupling`/`lugre`/`eso` must match the run that produced `sol`.
# =============================================================================
function compute_labels(sol, params, asmc, chi::Real,
                        coupling::Symbol, lugre, eso;
                        friction_fn, controller_fn, sawtooth_fn)
    N = length(sol.t)
    Vpx = zeros(4, N); Vpy = zeros(4, N); Wz  = zeros(4, N)
    Fxs = zeros(4, N); Fys = zeros(4, N); Mzs = zeros(4, N)
    Fpar = zeros(4, N); Fperp = zeros(4, N); Util = zeros(4, N)   # friction-circle coords
    Msw = zeros(4, N); Meq = zeros(4, N); Msat = zeros(4, N)
    Dhat_x   = zeros(N)                                 # MIMO DOB estimate δ̂_x(t)
    Dhat_y   = zeros(N)                                 # MIMO DOB estimate δ̂_y(t)
    Dhat_psi = zeros(N)                                 # MIMO DOB estimate δ̂_ψ(t)

    sdi = sin.(params.delta); cdi = cos.(params.delta); tdi = tan.(params.delta)
    px, py = params.wc_x, params.wc_y
    R, Rd = params.R, params.Ra

    @inbounds for k in 1:N
        t = sol.t[k]; u = sol.u[k]
        Vx, Vy, psi_dot, psi = u[1], u[2], u[3], u[4]
        ti = SVector(u[5], u[6], u[7], u[8])
        wi = SVector(u[9], u[10], u[11], u[12])
        gi = SVector(u[13], u[14], u[15], u[16])
        zx = SVector(u[22], u[23], u[24], u[25])
        zy = SVector(u[26], u[27], u[28], u[29])
        zs = SVector(u[30], u[31], u[32], u[33])

        ti_t  = sawtooth_fn.(ti)
        sti_t = sin.(ti_t); cti_t = cos.(ti_t); tti_t = tan.(ti_t)
        DYi   = Rd .* tdi .* tti_t

        Vpi_x = @. Vx - psi_dot * (py + DYi) - wi * R +
                   gi * sdi * (Rd * cti_t - R) + DYi * gi * cdi * sti_t
        Vpi_y = @. Vy + psi_dot * px +
                   gi * cdi * (R * cti_t - Rd)
        wzi   = @. psi_dot - gi * (-sti_t * cdi)

        for i in 1:4
            fx, fy, mz, _, _, _ = friction_fn(lugre, coupling, params.f_coulomb,
                                              params.N_per_roller[i], chi,
                                              wzi[i], Vpi_x[i], Vpi_y[i],
                                              zx[i], zy[i], zs[i])
            Fxs[i,k] = fx;  Fys[i,k] = fy;  Mzs[i,k] = mz
            # Roller-frame (friction-circle) decomposition + realized utilization:
            # F∥ along the drive direction (cosδ, sinδ), F⊥ along roller free-roll;
            # util_i = ‖F‖/(μ·N_i) is the realized master-gate ratio — directly
            # comparable to the 0.8-budget targets the profile TOMLs were gated on.
            Fpar[i,k]  =  fx * cdi[i] + fy * sdi[i]
            Fperp[i,k] = -fx * sdi[i] + fy * cdi[i]
            Util[i,k]  = hypot(fx, fy) / (params.f_coulomb * params.N_per_roller[i])
            Vpx[i,k] = Vpi_x[i]; Vpy[i,k] = Vpi_y[i]; Wz[i,k] = wzi[i]
        end

        Mi_sw, Mi_eq, _, dhat_vec = controller_fn(u, t, params, asmc, eso)
        Dhat_x[k]   = dhat_vec[1]
        Dhat_y[k]   = dhat_vec[2]
        Dhat_psi[k] = dhat_vec[3]
        for i in 1:4
            Msw[i,k] = Mi_sw[i]; Meq[i,k] = Mi_eq[i]
            Msat[i,k] = params.Max_torque * tanh((Mi_sw[i]+Mi_eq[i]) / params.Max_torque)
        end
    end
    return (; Vpx, Vpy, Wz, Fxs, Fys, Mzs, Fpar, Fperp, Util, Msw, Meq, Msat, Dhat_x, Dhat_y, Dhat_psi)
end

# =============================================================================
# Long-form DataFrame — one (profile, combo, mu, chi, friction_model) run.
#
# PINN-loader contract (data.py): the columns
#   Vx, Vy, psi_dot, w1..w4, theta1..theta4, Msat_1..4,
#   Fx_1..4, Fy_1..4, Mz_1..4, time
# are REQUIRED and must keep these exact names.
#
# Reference columns are now BODY-FRAME, read off the VelRef:
#   Vx_des, Vy_des  (ref.Vx/Vy),  psi_des (ref.psi),  omega_des (ref.Wz),
#   Ax_des, Ay_des  (ref.Ax/Ay),  alpha_des (ref.al)
# replacing the old world-frame Vxo_des/Vyo_des/xo_des/yo_des.
# =============================================================================
function assemble_dataframe(sol, labels, ref, meta::NamedTuple)
    N  = length(sol.t)
    st = hcat(sol.u...)                              # 39 × N
    df = DataFrame()
    df.profile        = fill(meta.profile, N)
    df.combo_idx      = fill(meta.combo_idx, N)
    df.friction_model = fill(String(meta.friction_model), N)
    df.mu             = fill(meta.mu, N)
    df.chi            = fill(meta.chi, N)
    df.time           = sol.t

    df.Vx, df.Vy, df.psi_dot, df.psi = st[1,:], st[2,:], st[3,:], st[4,:]
    for i in 1:4
        df[!, Symbol("theta$i")] = st[4+i, :]
        df[!, Symbol("w$i")]     = st[8+i, :]
        df[!, Symbol("gamma$i")] = st[12+i, :]
    end
    df.Kx, df.Ky, df.Kpsi = st[17,:], st[18,:], st[19,:]
    for i in 1:4
        df[!, Symbol("zx_$i")] = st[21+i, :]
        df[!, Symbol("zy_$i")] = st[25+i, :]
        df[!, Symbol("zs_$i")] = st[29+i, :]
    end
    df.Xo, df.Yo = st[20,:], st[21,:]

    # Reference channels — schema follows the reference kind (duck-typed so
    # this module stays independent of the Profiles types):
    #   VelRef (has :Vx) → body-frame:  Vx_des, Vy_des, psi_des, omega_des,
    #                                   Ax_des, Ay_des, alpha_des
    #   PosRef           → world-frame: xo_des, yo_des, Vxo_des, Vyo_des,
    #                                   psi_des, omega_des, alpha_des
    # PINN-required columns are unaffected either way; the loader ignores both
    # reference blocks.
    if hasproperty(ref, :Vx)
        df.Vx_des    = [ref.Vx(t)  for t in sol.t]
        df.Vy_des    = [ref.Vy(t)  for t in sol.t]
        df.psi_des   = [ref.psi(t) for t in sol.t]
        df.omega_des = [ref.Wz(t)  for t in sol.t]
        df.Ax_des    = [ref.Ax(t)  for t in sol.t]
        df.Ay_des    = [ref.Ay(t)  for t in sol.t]
        df.alpha_des = [ref.al(t)  for t in sol.t]
    else
        df.xo_des    = [ref.xo(t)  for t in sol.t]
        df.yo_des    = [ref.yo(t)  for t in sol.t]
        df.Vxo_des   = [ref.Vxo(t) for t in sol.t]
        df.Vyo_des   = [ref.Vyo(t) for t in sol.t]
        df.psi_des   = [ref.psi(t) for t in sol.t]
        df.omega_des = [ref.om(t)  for t in sol.t]
        df.alpha_des = [ref.al(t)  for t in sol.t]
    end

    for i in 1:4
        df[!, Symbol("Vpx_$i")]  = labels.Vpx[i, :]
        df[!, Symbol("Vpy_$i")]  = labels.Vpy[i, :]
        df[!, Symbol("wz_$i")]   = labels.Wz[i, :]
        df[!, Symbol("Fx_$i")]   = labels.Fxs[i, :]
        df[!, Symbol("Fy_$i")]   = labels.Fys[i, :]
        df[!, Symbol("Mz_$i")]   = labels.Mzs[i, :]
        df[!, Symbol("Fpar_$i")]  = labels.Fpar[i, :]
        df[!, Symbol("Fperp_$i")] = labels.Fperp[i, :]
        df[!, Symbol("util_$i")]  = labels.Util[i, :]
        df[!, Symbol("Msw_$i")]  = labels.Msw[i, :]
        df[!, Symbol("Meq_$i")]  = labels.Meq[i, :]
        df[!, Symbol("Msat_$i")] = labels.Msat[i, :]
    end
    df.Dhat_x   = labels.Dhat_x    # MIMO DOB estimate δ̂_x(t)
    df.Dhat_y   = labels.Dhat_y    # MIMO DOB estimate δ̂_y(t)
    df.Dhat_psi = labels.Dhat_psi  # MIMO DOB estimate δ̂_ψ(t)
    return df
end

# =============================================================================
# Disk writing — Arrow (zstd) + JLD2. CSV dropped (Arrow is the consumed
# format; CSV ~tripled write time/disk on long trajectories).
# JLD2 now also stores the resolved trajectory cfg + meta for full provenance:
# a saved run can be rebuilt exactly without consulting the TOMLs.
# =============================================================================
function write_outputs(df::DataFrame, sol, labels, params, asmc,
                       meta::NamedTuple; outdir::AbstractString,
                       cfg::Union{AbstractDict,Nothing} = nothing,
                       write_jld2::Bool = true)
    prefix     = output_prefix(outdir, meta)
    arrow_path = @sprintf("%s_chi_%.3f.arrow", prefix, meta.chi)
    jld2_path  = @sprintf("%s_chi_%.3f.jld2",  prefix, meta.chi)

    # ATOMIC writes: stream to a .tmp sibling, then rename. A job killed
    # mid-write leaves only .tmp debris (ignored by the loader and by the
    # driver's resume check) — never a truncated .arrow that resume would
    # mistake for a completed job. JLD2 first, Arrow last: the .arrow is the
    # resume marker, so it must be the final thing to appear.
    #
    # write_jld2 = false  -> Arrow only (the JLD2 sol_t/sol_u dump is ~1.3x the
    # Arrow size; skip it for large sweeps where the Arrow contract is all the
    # PINN consumes). The .arrow stays the sole resume marker either way.
    if write_jld2
        JLD2.jldsave(jld2_path * ".tmp"; sol_t = sol.t, sol_u = sol.u,
                     labels = labels,
                     params_nt = (; ntuple_from_params(params)...),
                     asmc_nt   = (; ntuple_from_asmc(asmc)...),
                     chi = meta.chi,
                     friction_model = String(meta.friction_model),
                     meta = meta,
                     traj_cfg = cfg === nothing ? Dict{String,Any}() : Dict{String,Any}(cfg))
        mv(jld2_path * ".tmp", jld2_path; force = true)
    end

    Arrow.write(arrow_path * ".tmp", df; compress = :zstd)
    mv(arrow_path * ".tmp", arrow_path; force = true)
    return (arrow = arrow_path, jld2 = write_jld2 ? jld2_path : nothing)
end

ntuple_from_params(p) = (h=p.h, l=p.l, R=p.R, Ra=p.Ra, m=p.m,
    m_wheel=p.m_wheel, J_wheel=p.J_wheel, J_roller=p.J_roller, ms=p.ms, Is=p.Is,
    p1_case1=p.p1_case1, p2_case1=p.p2_case1, p1_case2=p.p1_case2, p2_case2=p.p2_case2,
    f_coulomb=p.f_coulomb, N_total=p.N_total, rollers_per_wheel=p.rollers_per_wheel,
    delta=collect(p.delta), wc_x=collect(p.wc_x), wc_y=collect(p.wc_y),
    aX=p.aX, aY=p.aY, N_per_roller=collect(p.N_per_roller),
    M_inv=Matrix(p.M_inv), M_aug=Matrix(p.M_aug), M_aug_inv=Matrix(p.M_aug_inv),
    Max_torque=p.Max_torque)

ntuple_from_asmc(a) = (gamma_x=a.gamma_x, gamma_y=a.gamma_y, gamma_psi=a.gamma_psi,
    eps=a.eps, eps_psi=a.eps_psi, K_max_x=a.K_max_x, K_max_y=a.K_max_y, K_max_psi=a.K_max_psi,
    lam_x_min=a.lam_x_min, lam_x_max=a.lam_x_max, lam_y_min=a.lam_y_min, lam_y_max=a.lam_y_max,
    lam_psi_min=a.lam_psi_min, lam_psi_max=a.lam_psi_max, mu_xy=a.mu_xy, mu_psi=a.mu_psi)

# =============================================================================
# Streaming state logger — PresetTimeCallback + async writer task.
# Generalized: column count follows the actual state dimension (was hardcoded
# to 21; the state is 39-D now and may grow again).
# Returns (callback, close_fn). Call close_fn() after solve() to flush to disk.
# =============================================================================
function build_streaming_logger(path_arrow::AbstractString,
                                tstops::AbstractVector{<:Real})
    ch   = Channel{Tuple{Float64, Vector{Float64}}}(1024)
    rows = Vector{Vector{Float64}}()
    writer = @spawn begin
        for (t, u) in ch
            push!(rows, [t; u])
        end
    end
    affect! = integrator -> (put!(ch, (integrator.t, copy(integrator.u))); nothing)
    cb = PresetTimeCallback(collect(tstops), affect!)
    close_fn = () -> begin
        close(ch)
        wait(writer)
        n = length(rows)
        if n > 0
            D  = reduce(hcat, rows)                  # (1 + nstate) × N
            ns = size(D, 1) - 1
            df = DataFrame(time = D[1, :])
            for i in 1:ns
                df[!, Symbol("u$i")] = D[1+i, :]
            end
            Arrow.write(path_arrow, df; compress = :zstd)
        end
        return path_arrow
    end
    return cb, close_fn
end

# =============================================================================
# Reload helper — replaces the old notebook reload cell.
# =============================================================================
"""
    reload_run(paths) -> (df, jld)

`paths` is the NamedTuple returned by write_outputs. Reads the Arrow table
into a DataFrame and the JLD2 sidecar into a Dict.

Python-side equivalent:
    import pyarrow.feather as fe;  df = fe.read_feather("….arrow")
"""
function reload_run(paths::NamedTuple)
    df  = DataFrame(Arrow.Table(paths.arrow))
    jld = JLD2.load(paths.jld2)
    return df, jld
end

end # module DataStore
