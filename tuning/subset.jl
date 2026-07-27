# =============================================================================
# tuning/subset.jl — frozen curated trajectory subset for block tuning
# =============================================================================
module TuningSubsetMod

using JSON
using SHA

export TuningSubset, build_tuning_subset, save_subset_manifest, entries

"""
    TuningSubset

Frozen curated set of trajectories used for tuning any block (estimator,
controller, etc.).  `entries` is a vector of `(name, profile_toml, ref_type,
mu, config_dir, run_mode)` tuples.  `hash` is a deterministic content hash.
"""
struct TuningSubset
    entries::Vector{NamedTuple}
    hash::String
end

entries(ts::TuningSubset) = ts.entries

"""
    build_tuning_subset(run_dir, entries; include_optional=false) -> TuningSubset

Assemble the frozen CURATED tuning set from a fixed declared list spanning
excitation modes.  Each entry carries `ref_type` (`:velref` or `:posref`) and
`run_mode` (`:velocity` or `:pose`).  Serializes a manifest with a content hash.
The list is stable and deterministic.

  run_dir          :: AbstractString        a trajectory_files_run_* directory
  entries          :: Vector{NamedTuple}    the curated list
  include_optional :: Bool                  reserved for future optional entries
Returns `TuningSubset` (ordered entries + hash).
"""
function build_tuning_subset(run_dir::AbstractString,
                             entries::AbstractVector{<:NamedTuple};
                             include_optional::Bool=false)
    # Validate required fields and symbolic enums.
    required = (:name, :profile_toml, :ref_type, :mu, :config_dir, :run_mode)
    for e in entries
        for k in required
            haskey(e, k) || error("TuningSubset: entry missing field :$k in $e")
        end
        e.ref_type in (:velref, :posref) ||
            error("TuningSubset: ref_type must be :velref or :posref, got $(e.ref_type)")
        e.run_mode in (:velocity, :pose) ||
            error("TuningSubset: run_mode must be :velocity or :pose, got $(e.run_mode)")
        isfile(joinpath(e.config_dir, "profiles", e.profile_toml)) ||
            @warn "TuningSubset: profile file not found" path=joinpath(e.config_dir, "profiles", e.profile_toml)
    end
    # Sort by name for deterministic hashing.
    sorted = sort(entries, by = e -> e.name)
    # Build a plain JSON-serializable representation.  combo_idx (optional)
    # pins the profile combo; nothing = random pick per resolve_profile.
    data = [
        Dict("name"        => string(e.name),
             "profile_toml"=> string(e.profile_toml),
             "ref_type"    => string(e.ref_type),
             "mu"          => Float64(e.mu),
             "config_dir"  => string(e.config_dir),
             "run_mode"    => string(e.run_mode),
             "combo_idx"   => get(e, :combo_idx, nothing),
             "pose_fix_tier" => haskey(e, :pose_fix_tier) ? string(e.pose_fix_tier) : nothing)
        for e in sorted
    ]
    json_str = JSON.json(data)
    h = bytes2hex(sha256(json_str))
    return TuningSubset(sorted, h)
end

"""
    save_subset_manifest(subset::TuningSubset, outdir)

Write the subset manifest to `outdir/subset_manifest.json`.
Returns the written path.
"""
function save_subset_manifest(subset::TuningSubset, outdir::AbstractString)
    mkpath(outdir)
    path = joinpath(outdir, "subset_manifest.json")
    data = Dict(
        "hash"    => subset.hash,
        "entries" => [
            Dict("name"        => string(e.name),
                 "profile_toml"=> string(e.profile_toml),
                 "ref_type"    => string(e.ref_type),
                 "mu"          => Float64(e.mu),
                 "config_dir"  => string(e.config_dir),
                 "run_mode"    => string(e.run_mode),
                 "combo_idx"   => get(e, :combo_idx, nothing),
             "pose_fix_tier" => haskey(e, :pose_fix_tier) ? string(e.pose_fix_tier) : nothing)
            for e in subset.entries
        ]
    )
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
    return path
end

end # module
