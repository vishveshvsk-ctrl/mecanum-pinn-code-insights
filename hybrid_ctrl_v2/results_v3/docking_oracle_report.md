# Docking (step-hold) oracle eval — controller comparison in the preslip regime

Oracle feedback (clean + noisy seeds 101-105), campaign scoring (score3), converged v3 configs (0.9 archived + s100 clean-tuned), no re-tuning.

| config | traj | regime | score | tracking | chatter | ce |
|---|---|---|---|---|---|---|---|
| ASMC 0.9 ct | docking_a | clean | 0.18880 | 0.00074 | 0.36666 | 11.07 |
| ASMC 0.9 ct | docking_step | clean | 0.18113 | 0.00346 | 0.32225 | 15.68 |
| ASMC 0.9 ct | docking_a | noisy101 | 0.45329 | 0.07686 | 0.78133 | 11.92 |
| ASMC 0.9 ct | docking_step | noisy101 | 0.43445 | 0.05317 | 0.77119 | 16.44 |
| ASMC 0.9 ct | docking_a | noisy102 | 0.42985 | 0.05343 | 0.78114 | 11.95 |
| ASMC 0.9 ct | docking_step | noisy102 | 0.41717 | 0.03605 | 0.77074 | 16.46 |
| ASMC 0.9 ct | docking_a | noisy103 | 0.45257 | 0.07588 | 0.78176 | 11.95 |
| ASMC 0.9 ct | docking_step | noisy103 | 0.43334 | 0.05198 | 0.77087 | 16.54 |
| ASMC 0.9 ct | docking_a | noisy104 | 0.39931 | 0.02271 | 0.78192 | 11.87 |
| ASMC 0.9 ct | docking_step | noisy104 | 0.40023 | 0.01909 | 0.77088 | 16.43 |
| ASMC 0.9 ct | docking_a | noisy105 | 0.48069 | 0.10441 | 0.78093 | 11.93 |
| ASMC 0.9 ct | docking_step | noisy105 | 0.45617 | 0.07426 | 0.77272 | 16.41 |
| PID-CT 0.9 ct | docking_a | clean | 0.17140 | 0.00032 | 0.32860 | 11.14 |
| PID-CT 0.9 ct | docking_step | clean | 0.18735 | 0.00195 | 0.33942 | 15.68 |
| PID-CT 0.9 ct | docking_a | noisy101 | 0.44116 | 0.07140 | 0.76750 | 11.71 |
| PID-CT 0.9 ct | docking_step | noisy101 | 0.42196 | 0.04839 | 0.75462 | 16.31 |
| PID-CT 0.9 ct | docking_a | noisy102 | 0.41433 | 0.04451 | 0.76793 | 11.64 |
| PID-CT 0.9 ct | docking_step | noisy102 | 0.40609 | 0.03341 | 0.75299 | 16.24 |
| PID-CT 0.9 ct | docking_a | noisy103 | 0.44716 | 0.07752 | 0.76721 | 11.71 |
| PID-CT 0.9 ct | docking_step | noisy103 | 0.42446 | 0.05157 | 0.75321 | 16.30 |
| PID-CT 0.9 ct | docking_a | noisy104 | 0.39714 | 0.02732 | 0.76738 | 11.76 |
| PID-CT 0.9 ct | docking_step | noisy104 | 0.39203 | 0.01883 | 0.75393 | 16.29 |
| PID-CT 0.9 ct | docking_a | noisy105 | 0.47439 | 0.10419 | 0.76791 | 11.83 |
| PID-CT 0.9 ct | docking_step | noisy105 | 0.44496 | 0.07188 | 0.75338 | 16.35 |
| ASMC s100 ct | docking_a | clean | 0.18847 | 0.00088 | 0.36559 | 11.07 |
| ASMC s100 ct | docking_step | clean | 0.18683 | 0.00343 | 0.33508 | 15.66 |
| ASMC s100 ct | docking_a | noisy101 | 0.45343 | 0.07719 | 0.78164 | 11.76 |
| ASMC s100 ct | docking_step | noisy101 | 0.43409 | 0.05312 | 0.77103 | 16.33 |
| ASMC s100 ct | docking_a | noisy102 | 0.42757 | 0.05128 | 0.78136 | 11.85 |
| ASMC s100 ct | docking_step | noisy102 | 0.41766 | 0.03648 | 0.77107 | 16.42 |
| ASMC s100 ct | docking_a | noisy103 | 0.45312 | 0.07650 | 0.78223 | 11.82 |
| ASMC s100 ct | docking_step | noisy103 | 0.43081 | 0.04995 | 0.77042 | 16.40 |
| ASMC s100 ct | docking_a | noisy104 | 0.40253 | 0.02635 | 0.78154 | 11.75 |
| ASMC s100 ct | docking_step | noisy104 | 0.40154 | 0.02062 | 0.77077 | 16.35 |
| ASMC s100 ct | docking_a | noisy105 | 0.48082 | 0.10455 | 0.78125 | 11.86 |
| ASMC s100 ct | docking_step | noisy105 | 0.45579 | 0.07428 | 0.77227 | 16.32 |
| PID-CT s100 ct | docking_a | clean | 0.17360 | 0.00057 | 0.33296 | 11.13 |
| PID-CT s100 ct | docking_step | clean | 0.17184 | 0.00323 | 0.30208 | 15.68 |
| PID-CT s100 ct | docking_a | noisy101 | 0.42293 | 0.05897 | 0.75571 | 11.47 |
| PID-CT s100 ct | docking_step | noisy101 | 0.40785 | 0.04043 | 0.74216 | 16.05 |
| PID-CT s100 ct | docking_a | noisy102 | 0.40644 | 0.04239 | 0.75593 | 11.47 |
| PID-CT s100 ct | docking_step | noisy102 | 0.40381 | 0.03717 | 0.74100 | 15.93 |
| PID-CT s100 ct | docking_a | noisy103 | 0.43824 | 0.07402 | 0.75627 | 11.47 |
| PID-CT s100 ct | docking_step | noisy103 | 0.41820 | 0.05008 | 0.74343 | 16.11 |
| PID-CT s100 ct | docking_a | noisy104 | 0.39234 | 0.02848 | 0.75499 | 11.57 |
| PID-CT s100 ct | docking_step | noisy104 | 0.38724 | 0.01937 | 0.74320 | 16.05 |
| PID-CT s100 ct | docking_a | noisy105 | 0.46930 | 0.10495 | 0.75558 | 11.68 |
| PID-CT s100 ct | docking_step | noisy105 | 0.43573 | 0.06841 | 0.74079 | 16.30 |

## Per-config mean (both trajectories, noise seeds averaged)

| config | traj | score | tracking | chatter |
|---|---|---|---|---|
| ASMC 0.9 ct | docking_a | 0.44314 | 0.06666 | 0.78142 |
| ASMC 0.9 ct | docking_step | 0.42827 | 0.04691 | 0.77128 |
| PID-CT 0.9 ct | docking_a | 0.43484 | 0.06499 | 0.76759 |
| PID-CT 0.9 ct | docking_step | 0.41790 | 0.04482 | 0.75362 |
| ASMC s100 ct | docking_a | 0.44349 | 0.06717 | 0.78160 |
| ASMC s100 ct | docking_step | 0.42798 | 0.04689 | 0.77111 |
| PID-CT s100 ct | docking_a | 0.42585 | 0.06176 | 0.75570 |
| PID-CT s100 ct | docking_step | 0.41057 | 0.04309 | 0.74212 |
