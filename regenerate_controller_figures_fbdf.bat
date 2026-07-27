@echo off
REM Regenerate all controller comparison figures using the FBDF solver.
REM Overwrites existing trace and figure folders.

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"
set "PYTHON_EXE=C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe"

cd /d "%ROOT%"

echo === Saving ESKF traces (4 evaluation trajs) ===
"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\save_controller_eskf_traces.jl ^
  --estimator eskf ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_traces

echo === Saving oracle traces (4 evaluation trajs) ===
"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\save_controller_eskf_traces.jl ^
  --estimator oracle ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_traces

echo === Saving ESKF nw0 traces (octagon combo 1, n_waves=0) ===
"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\save_controller_eskf_traces.jl ^
  --estimator eskf ^
  --traj-spec octagon_nw0:octagon_mu_0p5.toml:1 ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_traces_nw0

echo === Saving oracle nw0 traces (octagon combo 1, n_waves=0) ===
"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\save_controller_eskf_traces.jl ^
  --estimator oracle ^
  --traj-spec octagon_nw0:octagon_mu_0p5.toml:1 ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_traces_nw0

echo === Plotting ESKF figures ===
"%PYTHON_EXE%" hybrid_ctrl\estimator_tuning\plot_controller_eskf_traces.py ^
  --trace-dir hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_traces ^
  --out-dir hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_figures ^
  --feedback eskf

echo === Plotting oracle figures ===
"%PYTHON_EXE%" hybrid_ctrl\estimator_tuning\plot_controller_eskf_traces.py ^
  --trace-dir hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_traces ^
  --out-dir hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_figures ^
  --feedback oracle

echo === Plotting ESKF nw0 figures ===
"%PYTHON_EXE%" hybrid_ctrl\estimator_tuning\plot_controller_eskf_traces.py ^
  --trace-dir hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_traces_nw0 ^
  --out-dir hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_figures_nw0 ^
  --feedback eskf

echo === Plotting oracle nw0 figures ===
"%PYTHON_EXE%" hybrid_ctrl\estimator_tuning\plot_controller_eskf_traces.py ^
  --trace-dir hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_traces_nw0 ^
  --out-dir hybrid_ctrl\estimator_tuning\reports\controller_oracle_pose_figures_nw0 ^
  --feedback oracle

echo === All figures regenerated with FBDF ===
