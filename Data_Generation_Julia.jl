# ============================================================================
# Data_Generation_Julia.jl — profile-based parallel sweep driver (Step 4).
#
# Replaces the (beta, amplitude) grid driver. Enumerates EVERY combo row of the
# selected profile TOMLs (Profiles.enumerate_jobs), builds each reference once
# (single-threaded, publish! → CURRENT_REF/CURRENT_POSREF), then fans the
# physics combos (mu × chi × friction_model from base.toml) across threads.
#
#   julia --project=. -t auto Data_Generation_Julia.jl
#   julia --project=. -t 8    Data_Generation_Julia.jl --dry-run
#   julia --project=. -t 8    Data_Generation_Julia.jl --profiles octagon.toml,ellipse.toml
#   julia --project=. -t 8    Data_Generation_Julia.jl --timeout 900   # straggler retry pass
#
# THREADING MODEL (unchanged philosophy, new mechanism):
#   outer  — sequential over trajectories; build_job + publish! mutate the
#            module-level reference slots and MUST be single-threaded.
#   inner  — Threads.@threads :static over (mu, chi, fm); threads only READ the
#            published reference; PlatformParams/ODEProblem/solve all local.
#
# RESUME: a job is complete iff its .arrow exists (DataStore.expected_output —
# the SAME function the writer uses; atomic tmp→rename writes mean no partials).
# Timed-out / failed jobs leave no .arrow and are retried on the next
# invocation — typically with a longer --timeout. build_job is deterministic
# (sweep keys from hash((sweep_seed, profile, combo))), so a retry reproduces
# the IDENTICAL trajectory.
#
# CONFIG: everything physics/solver comes from <config-dir>/base.toml:
#   [physics]  mu_friction, chi, friction_model, friction_case — scalar OR list
#              (lists are swept; scalars are a single point)
#   [solver]   name ("TRBDF2"|"Rodas5P"|"RadauIIA5"|"FBDF"|"KenCarp47"|"QNDF"),
#              reltol, dtmax, maxiters, saveat_rate
#   [solver.abstol] + [solver.abstol_counts]  → per-group 39-vector (canonical
#              state order asserted below)
#   [dob]      OPTIONAL observer table (enable, omega_o_*, kind, k1/k2_*,
#              eps_obs); defaults below = the notebook's production values.
# The solver-benchmark chat's winner lands here as a base.toml edit — no driver
# change needed.
# ============================================================================

using Dates
using Printf

# -------------------------- CLI parsing --------------------------
function parse_cli(args)
    cfg = Dict{Symbol,Any}(
        :dry_run    => false,
        :script     => "run_one.jl",
        :config_dir => "trajectory_files",
        :outdir     => "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
        :profiles   => String[],          # empty ⇒ discover all
        :resume     => true,
        :timeout    => 300.0,
        :progress   => 30.0,              # seconds between per-thread progress lines (Inf = off)
        :sweep_seed => 1234,
        :limit      => typemax(Int),      # cap on trajectory jobs (smoke runs)
        :combos     => Int[],             # restrict to these combo_idx (empty ⇒ all); for subset runs
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--dry-run";            cfg[:dry_run] = true;                          i += 1
        elseif a == "--script";             cfg[:script] = args[i+1];                      i += 2
        elseif a == "--config-dir";         cfg[:config_dir] = args[i+1];                  i += 2
        elseif a == "--outdir";             cfg[:outdir] = args[i+1];                      i += 2
        elseif a == "--profiles";           cfg[:profiles] = String.(split(args[i+1], ","));i += 2
        elseif a == "--no-resume";          cfg[:resume] = false;                          i += 1
        elseif a == "--timeout";            cfg[:timeout] = parse(Float64, args[i+1]);     i += 2
        elseif a == "--no-timeout";         cfg[:timeout] = Inf;                           i += 1
        elseif a == "--progress-interval";  cfg[:progress] = parse(Float64, args[i+1]);    i += 2
        elseif a == "--no-progress";        cfg[:progress] = Inf;                          i += 1
        elseif a == "--sweep-seed";         cfg[:sweep_seed] = parse(Int, args[i+1]);      i += 2
        elseif a == "--limit";              cfg[:limit] = parse(Int, args[i+1]);           i += 2
        elseif a == "--combos";             cfg[:combos] = parse.(Int, split(args[i+1], ","));i += 2
        elseif a in ("-h", "--help")
            println("""
            Profile-based Mecanum data-generation sweep.

              julia --project=. -t auto Data_Generation_Julia.jl [options]

              --config-dir PATH        trajectory config root (default: trajectory_files)
              --profiles a.toml,b.toml subset of profile TOMLs (default: ALL in
                                       <config-dir>/profiles or flat beside base.toml)
              --script PATH            extracted simulator (default: run_one.jl)
              --outdir PATH            output directory
              --sweep-seed N           seed for [profile.sweep] draws (default 1234;
                                       MUST match across resume passes)
              --timeout N              wall-clock seconds per solve (default 300)
              --no-timeout             disable per-solve timeout
              --progress-interval N    seconds between per-thread progress lines (default 30)
              --no-progress            disable progress lines
              --limit N                only the first N trajectory jobs (smoke runs)
              --dry-run                print the plan (resume-aware) and exit
              --no-resume              re-run even if outputs exist
            """)
            exit(0)
        else
            error("Unknown argument: $a (try --help)")
        end
    end
    return cfg
end

const CFG = parse_cli(ARGS)
const TIMEOUT_SECONDS   = CFG[:timeout]
const PROGRESS_INTERVAL = CFG[:progress]

# -------------------------- Load simulator + modules --------------------------
@info "Loading simulator (first load compiles — takes 30–90 s)…" threads=Threads.nthreads()
include(CFG[:script])     # physics, controllers, dynamics, build_initial_state;
                          # also includes profiles.jl + datastore.jl via the
                          # notebook's includes cell (Profiles, DataStore in scope)

if Threads.nthreads() == 1
    @warn "Running with 1 thread. Restart with `julia -t auto` (or -t N) for the parallel inner loop."
end

# -------------------------- base.toml → sweep + solver config ----------------
import TOML
const BASE = Profiles.load_base(CFG[:config_dir])

_aslist(x) = x isa AbstractVector ? collect(x) : [x]

const PHYS            = get(BASE, "physics", Dict{String,Any}())
const MU_VALUES       = Float64.(_aslist(get(PHYS, "mu_friction", 0.5)))
const CHI_VALUES      = Float64.(_aslist(get(PHYS, "chi", 0.005)))
const FM_VALUES       = Symbol.(_aslist(get(PHYS, "friction_model", "lugre_adamov")))
const FRICTION_CASE   = Int(get(PHYS, "friction_case", 1))

const SOLV = get(BASE, "solver", Dict{String,Any}())
const SOLVER_MAP = Dict(
    "TRBDF2" => () -> TRBDF2(),   "Rodas5P"  => () -> Rodas5P(),
    "RadauIIA5" => () -> RadauIIA5(), "FBDF" => () -> FBDF(),
    "KenCarp47" => () -> KenCarp47(), "QNDF" => () -> QNDF(),
)
const SOLVER_NAME = String(get(SOLV, "name", "TRBDF2"))
haskey(SOLVER_MAP, SOLVER_NAME) ||
    error("base.toml [solver].name = \"$SOLVER_NAME\" not in $(sort(collect(keys(SOLVER_MAP))))")
const RELTOL      = Float64(get(SOLV, "reltol", 1e-8))
const DTMAX       = Float64(get(SOLV, "dtmax", 0.001))
const MAXITERS    = Int(get(SOLV, "maxiters", 10^7))
const SAVEAT_RATE = Float64(get(SOLV, "saveat_rate", 2000))

"""Rebuild the 39-vector ABSTOL from [solver.abstol] + [solver.abstol_counts].
The CANONICAL group order is the state layout — TOML tables don't preserve
order, so it is hardcoded here and the counts must sum to the state dimension."""
function build_abstol(base)
    order = ["body_vel","psi","wtheta","womega","gamma","gains",
             "worldpos","bristle","bristle_rot","observer","dist_est"]
    at  = base["solver"]["abstol"]
    cnt = base["solver"]["abstol_counts"]
    for g in order
        haskey(at, g)  || error("base.toml [solver.abstol] missing group '$g'")
        haskey(cnt, g) || error("base.toml [solver.abstol_counts] missing group '$g'")
    end
    v = vcat((fill(Float64(at[g]), Int(cnt[g])) for g in order)...)
    length(v) == 39 || error("ABSTOL groups sum to $(length(v)) ≠ 39 — state layout changed?")
    return v
end
const ABSTOL_VEC = build_abstol(BASE)

# DOB / observer — [dob] table optional; defaults = notebook production values.
const DOBC = get(BASE, "dob", Dict{String,Any}())
const ESO = ESOParams(kind = Symbol(get(DOBC, "kind", "super_twisting")),
                      omega_o_x   = Float64(get(DOBC, "omega_o_x", 0.0)),
                      omega_o_y   = Float64(get(DOBC, "omega_o_y", 0.0)),
                      omega_o_psi = Float64(get(DOBC, "omega_o_psi", 6π)),
                      k1_x = Float64(get(DOBC, "k1_x", 0.0)),  k2_x = Float64(get(DOBC, "k2_x", 0.0)),
                      k1_y = Float64(get(DOBC, "k1_y", 0.0)),  k2_y = Float64(get(DOBC, "k2_y", 0.0)),
                      k1_psi = Float64(get(DOBC, "k1_psi", 15.0)),
                      k2_psi = Float64(get(DOBC, "k2_psi", 80.0)),
                      eps_obs = Float64(get(DOBC, "eps_obs", 1e-2)),
                      enable  = Bool(get(DOBC, "enable", true)))

# ASMCParams' gamma_y/gamma_psi defaults read the Main global `use_dob` (set to a
# stand-in in run_one.jl). Bind it to the observer's actual enable state so the
# controller's DOB-dependent adaptation gains track base.toml [dob].enable — the
# single source of truth — instead of run_one.jl's stand-in default.
use_dob = ESO.enable

# -------------------------- Job enumeration --------------------------
function discover_profiles(config_dir)
    for d in (joinpath(config_dir, "profiles"), config_dir)
        isdir(d) || continue
        fs = sort(filter(f -> endswith(f, ".toml") && f != "base.toml", readdir(d)))
        isempty(fs) || return fs
    end
    error("No profile TOMLs found under $config_dir (looked in profiles/ and flat)")
end

const PROFILE_FILES = isempty(CFG[:profiles]) ? discover_profiles(CFG[:config_dir]) : CFG[:profiles]
traj_jobs = Profiles.enumerate_jobs(CFG[:config_dir], PROFILE_FILES; sweep_seed = CFG[:sweep_seed])
# --combos: restrict to specific combo_idx (subset runs). Full combos arrays are kept
# in the TOML so combo_idx stays the original index ⇒ matched to existing files.
if !isempty(CFG[:combos])
    want = Set(CFG[:combos])
    traj_jobs = filter(tj -> tj.combo_idx in want, traj_jobs)
end
length(traj_jobs) > CFG[:limit] && (traj_jobs = traj_jobs[1:CFG[:limit]])

inner_combos = [(mu = mu, chi = chi, fm = fm)
                for mu in MU_VALUES for chi in CHI_VALUES for fm in FM_VALUES]

make_meta(tj, ic) = (profile = tj.profile, combo_idx = tj.combo_idx,
                     mu = ic.mu, chi = ic.chi,
                     friction_case = FRICTION_CASE, friction_model = ic.fm,
                     sweep_seed = tj.sweep_seed)

const OUT = CFG[:outdir]
mkpath(OUT)

n_traj  = length(traj_jobs)
n_inner = length(inner_combos)
n_total = n_traj * n_inner
@info "Sweep plan" profiles=join(PROFILE_FILES, ", ") trajectories=n_traj inner_per_traj=n_inner total=n_total solver=SOLVER_NAME dtmax=DTMAX saveat_hz=SAVEAT_RATE

# Wrapped in a function so the accumulator loop shares one local scope — a
# top-level `for` introduces soft scope, under which `pend += 1` would be read
# as an undefined local (Julia ≥1.5 script semantics).
function dry_run_report()
    pend = 0; done = 0
    per = Dict{String,Vector{Int}}()   # profile → [pending, done]
    for tj in traj_jobs, ic in inner_combos
        m = make_meta(tj, ic)
        d = isfile(DataStore.expected_output(OUT, m))
        v = get!(per, tj.profile, [0, 0]);  v[d ? 2 : 1] += 1
        d ? (done += 1) : (pend += 1)
    end
    for (p, v) in sort(collect(per))
        @printf("  %-28s pending %5d   done %5d\n", p, v[1], v[2])
    end
    @printf("  %-28s pending %5d   done %5d   (total %d)\n", "TOTAL", pend, done, n_total)
end

if CFG[:dry_run]
    dry_run_report()
    exit(0)
end

# -------------------------- JSONL job log --------------------------
"""Minimal dependency-free JSON-object emitter for a flat NamedTuple."""
function JSON_line(nt::NamedTuple)
    esc(s::AbstractString) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"",
                                            "\n" => "\\n", "\r" => "\\r", "\t" => "\\t") * "\""
    fmt(v::AbstractString) = esc(v)
    fmt(v::Bool)           = v ? "true" : "false"
    fmt(v::Integer)        = string(v)
    fmt(v::AbstractFloat)  = isfinite(v) ? string(v) : "null"
    fmt(v::Nothing)        = "null"
    fmt(v)                 = esc(string(v))
    return "{" * join([string(esc(string(k)), ":", fmt(v)) for (k, v) in pairs(nt)], ",") * "}"
end

const JOBS_LOG_PATH = joinpath(OUT, "jobs_log_profiles.jsonl")
const RUN_ID = Dates.format(now(), "yyyymmdd_HHMMSS")
jobs_log_lock = ReentrantLock()

function log_job!(meta, status::AbstractString; wall_s::Real = NaN,
                  sim_t::Real = NaN, sim_T::Real = NaN,
                  retcode::AbstractString = "", err_msg::AbstractString = "")
    rec = (; event = "job", run_id = RUN_ID, status = String(status),
           profile = meta.profile, combo_idx = meta.combo_idx,
           mu = meta.mu, chi = meta.chi,
           friction_case = meta.friction_case, friction_model = String(meta.friction_model),
           wall_s = Float64(wall_s), sim_t = Float64(sim_t), sim_T = Float64(sim_T),
           retcode = String(retcode), err_msg = String(err_msg),
           thread = Threads.threadid(), timestamp = string(now()))
    lock(jobs_log_lock) do
        open(JOBS_LOG_PATH, "a") do io; println(io, JSON_line(rec)); end
    end
end

open(JOBS_LOG_PATH, "a") do io
    println(io, JSON_line((; event="run_start", run_id=RUN_ID, timestamp=string(now()),
        profiles=join(PROFILE_FILES, ","), nthreads=Threads.nthreads(),
        total_jobs=n_total, timeout_s=TIMEOUT_SECONDS,
        solver=SOLVER_NAME, reltol=RELTOL, dtmax=DTMAX, saveat_rate=SAVEAT_RATE,
        sweep_seed=CFG[:sweep_seed], outdir=OUT)))
end

# -------------------------- Main sweep --------------------------
t_start    = now()
ok_count   = Threads.Atomic{Int}(0)
err_count  = Threads.Atomic{Int}(0)
skip_count = Threads.Atomic{Int}(0)
const MAX_ERR_REPORT = 5
errors_seen = Vector{Any}();  errors_lock = ReentrantLock()

for (ti, tj) in enumerate(traj_jobs)
    # ---- Build + publish the reference: SINGLE-THREADED, once per trajectory.
    ref, traj_cfg = Profiles.build_job(tj)
    Profiles.publish!(ref)                      # VelRef→CURRENT_REF, PosRef→CURRENT_POSREF
    ctrl   = Profiles.is_velref() ? asmc_torques_vel : asmc_torques
    T      = ref.T_total
    tstops = ref.tstops

    # Boundary-layer width ∝ commanded speed scale (kind-aware peak |V|).
    PEAK_V = let ts = range(0, T; length = 2001)
        hasproperty(ref, :Vx) ? maximum(hypot(ref.Vx(t),  ref.Vy(t))  for t in ts) :
                                maximum(hypot(ref.Vxo(t), ref.Vyo(t)) for t in ts)
    end
    asmc = ASMCParams(eps = max(0.025 * PEAK_V, 0.001))

    @info "[$ti/$n_traj] trajectory" profile=tj.profile combo=tj.combo_idx kind=(Profiles.is_velref() ? "VelRef" : "PosRef") T=round(T; digits=1) peakV=round(PEAK_V; digits=2)
    ti == 1 && @info "  First trajectory: expect 1–5 min of compile silence for the stiff solver."
    flush(stdout); flush(stderr)

    t_eval = collect(range(0.0, T; length = round(Int, T * SAVEAT_RATE) + 1))

    # ---- Inner: (mu, chi, fm) fanout — threads only READ the published ref.
    Threads.@threads :static for ic in inner_combos
        meta = make_meta(tj, ic)
        out_path = DataStore.expected_output(OUT, meta)
        if CFG[:resume] && isfile(out_path)
            Threads.atomic_add!(skip_count, 1)
            log_job!(meta, "skip")
            continue
        end

        t0 = time()
        try
            params = PlatformParams(BASE; mu_friction = ic.mu)
            p1, p2 = FRICTION_CASE == 1 ? (params.p1_case1, params.p2_case1) :
                                           (params.p1_case2, params.p2_case2)
            u0   = build_initial_state(params, asmc)
            prob = ODEProblem(dynamics_full_mf_asmc!, u0, (0.0, T),
                              (params, asmc, ic.chi, p1, p2,
                               coupling_of(ic.fm), lugre, ESO, ctrl))

            # Wall-clock deadline: per-step DiscreteCallback (NOT PeriodicCallback —
            # when dt collapses, sim-time-based callbacks never fire; a per-step
            # condition always does, so the timeout stays enforceable).
            t_solve = time();  timed_out = Ref(false)
            tcond(u, t, integ) = (time() - t_solve) >= TIMEOUT_SECONDS
            taffect!(integ) = (timed_out[] = true; terminate!(integ); nothing)
            timeout_cb = DiscreteCallback(tcond, taffect!; save_positions = (false, false))

            # Per-thread, wall-clock-throttled progress line (informational only).
            last_print = Ref(time());  tid = Threads.threadid()
            function prog!(integ)
                tnow = time()
                if tnow - last_print[] >= PROGRESS_INTERVAL
                    @info @sprintf("[traj %d | th %2d] %s c%03d mu=%.2f chi=%.4f %s  sim=%6.2f/%.1fs  dt=%.2e  wall=%.0fs",
                                   ti, tid, tj.profile, tj.combo_idx, ic.mu, ic.chi,
                                   String(ic.fm), integ.t, T,
                                   (try integ.dt catch; NaN end), tnow - t_solve)
                    flush(stderr);  last_print[] = tnow
                end
                return nothing
            end

            cbs = Any[]
            isfinite(TIMEOUT_SECONDS)   && push!(cbs, timeout_cb)
            isfinite(PROGRESS_INTERVAL) && push!(cbs, PeriodicCallback(prog!, 1.0; initial_affect = false))

            sol = solve(prob, SOLVER_MAP[SOLVER_NAME]();
                        reltol = RELTOL, abstol = ABSTOL_VEC,
                        saveat = t_eval, tstops = tstops,
                        dtmax = DTMAX, dtmin = 1e-10, force_dtmin = true,
                        maxiters = MAXITERS,
                        callback = isempty(cbs) ? nothing : CallbackSet(cbs...))

            terminated = timed_out[] || sol.retcode == ReturnCode.Terminated
            success    = sol.retcode == ReturnCode.Success

            if terminated
                dt = time() - t0
                Threads.atomic_add!(err_count, 1)
                lock(errors_lock) do
                    length(errors_seen) < MAX_ERR_REPORT &&
                        push!(errors_seen, (meta = meta, err = "TIMEOUT after $(round(Int, dt))s at sim t=$(round(sol.t[end]; digits=2))s"))
                end
                log_job!(meta, "timeout"; wall_s = dt, sim_t = sol.t[end], sim_T = T,
                         retcode = string(sol.retcode), err_msg = "TIMEOUT after $(round(Int, dt))s")
            elseif !success
                dt = time() - t0
                Threads.atomic_add!(err_count, 1)
                lock(errors_lock) do
                    length(errors_seen) < MAX_ERR_REPORT &&
                        push!(errors_seen, (meta = meta, err = "retcode=$(sol.retcode)"))
                end
                log_job!(meta, "solver_error"; wall_s = dt, sim_t = sol.t[end], sim_T = T,
                         retcode = string(sol.retcode))
            else
                labels = DataStore.compute_labels(sol, params, asmc, ic.chi,
                                                  coupling_of(ic.fm), lugre, ESO;
                                                  friction_fn   = lugre_dyn_rates,
                                                  controller_fn = ctrl,
                                                  sawtooth_fn   = sawtooth_approx)
                df = DataStore.assemble_dataframe(sol, labels, ref, meta)
                DataStore.write_outputs(df, sol, labels, params, asmc, meta;
                                        outdir = OUT, cfg = traj_cfg,
                                        write_jld2 = false)   # Arrow-only: JLD2 sidecar is storage-heavy
                dt = time() - t0
                Threads.atomic_add!(ok_count, 1)
                log_job!(meta, "ok"; wall_s = dt, sim_t = T, sim_T = T,
                         retcode = string(sol.retcode))
                @info @sprintf("[traj %d/%d | th %2d] ok  %s c%03d mu=%.2f chi=%.4f %s  %.1fs",
                               ti, n_traj, Threads.threadid(), tj.profile, tj.combo_idx,
                               ic.mu, ic.chi, String(ic.fm), dt)
            end
        catch e
            dt = time() - t0
            Threads.atomic_add!(err_count, 1)
            msg = sprint(showerror, e)
            lock(errors_lock) do
                length(errors_seen) < MAX_ERR_REPORT && push!(errors_seen, (meta = meta, err = msg[1:min(end, 300)]))
            end
            log_job!(meta, "exception"; wall_s = dt, err_msg = msg[1:min(end, 500)])
            @warn "[traj $ti] exception" profile=tj.profile combo=tj.combo_idx mu=ic.mu chi=ic.chi err=msg[1:min(end, 200)]
        end
    end
end

elapsed = canonicalize(now() - t_start)
@info "Sweep complete" ok=ok_count[] failed=err_count[] skipped=skip_count[] elapsed=elapsed
if !isempty(errors_seen)
    @warn "First $(length(errors_seen)) failures:"
    for e in errors_seen
        @warn "  $(e.meta.profile) c$(e.meta.combo_idx) mu=$(e.meta.mu) chi=$(e.meta.chi): $(e.err)"
    end
    @info "Retry stragglers with a longer budget:  julia --project=. -t auto Data_Generation_Julia.jl --timeout 900"
end
@info "Per-job log" path=JOBS_LOG_PATH run_id=RUN_ID
