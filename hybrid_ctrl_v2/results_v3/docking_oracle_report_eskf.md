# Docking (step-hold) ESKF eval — estimator performance in the preslip regime

Frozen tuned ESKF v4 (seed-4 config) + realistic sensor suite (pose_fix_tier=:docking), noise seeds 101-105, campaign scoring (score3).

| config | traj | regime | score | tracking | chatter | est_pos [m] | est_vel [m/s] |
|---|---|---|---|---|---|---|---|
| ASMC 0.9 ct | docking_a | eskf1 | 0.29173 | 0.00515 | 0.58157 | 0.00304 | 0.00256 |
| ASMC 0.9 ct | docking_step | eskf1 | 0.30092 | 0.00786 | 0.57512 | 0.00306 | 0.00312 |
| ASMC 0.9 ct | docking_a | eskf2 | 0.30012 | 0.00936 | 0.59181 | 0.00349 | 0.00258 |
| ASMC 0.9 ct | docking_step | eskf2 | 0.30721 | 0.00877 | 0.58840 | 0.00347 | 0.00225 |
| ASMC 0.9 ct | docking_a | eskf3 | 0.29925 | 0.01867 | 0.57047 | 0.00338 | 0.00174 |
| ASMC 0.9 ct | docking_step | eskf3 | 0.29978 | 0.01346 | 0.56175 | 0.00338 | 0.00160 |
| ASMC 0.9 ct | docking_a | eskf4 | 0.30411 | 0.01396 | 0.59063 | 0.00352 | 0.00189 |
| ASMC 0.9 ct | docking_step | eskf4 | 0.31137 | 0.01402 | 0.58587 | 0.00353 | 0.00241 |
| ASMC 0.9 ct | docking_a | eskf5 | 0.29641 | 0.01549 | 0.57041 | 0.00337 | 0.00260 |
| ASMC 0.9 ct | docking_step | eskf5 | 0.30146 | 0.01324 | 0.56563 | 0.00336 | 0.00231 |
| PID-CT 0.9 ct | docking_a | eskf1 | 0.29021 | 0.01081 | 0.56607 | 0.00306 | 0.00363 |
| PID-CT 0.9 ct | docking_step | eskf1 | 0.29244 | 0.00734 | 0.55796 | 0.00306 | 0.00408 |
| PID-CT 0.9 ct | docking_a | eskf2 | 0.29838 | 0.01275 | 0.58178 | 0.00350 | 0.00292 |
| PID-CT 0.9 ct | docking_step | eskf2 | 0.30180 | 0.01100 | 0.57223 | 0.00348 | 0.00248 |
| PID-CT 0.9 ct | docking_a | eskf3 | 0.27807 | 0.00797 | 0.54788 | 0.00338 | 0.00163 |
| PID-CT 0.9 ct | docking_step | eskf3 | 0.28143 | 0.00654 | 0.53729 | 0.00338 | 0.00138 |
| PID-CT 0.9 ct | docking_a | eskf4 | 0.29276 | 0.00826 | 0.57969 | 0.00351 | 0.00320 |
| PID-CT 0.9 ct | docking_step | eskf4 | 0.29744 | 0.00588 | 0.57426 | 0.00353 | 0.00293 |
| PID-CT 0.9 ct | docking_a | eskf5 | 0.28177 | 0.01086 | 0.54879 | 0.00339 | 0.00315 |
| PID-CT 0.9 ct | docking_step | eskf5 | 0.28535 | 0.00725 | 0.54380 | 0.00339 | 0.00331 |
| ASMC s100 ct | docking_a | eskf1 | 0.28979 | 0.00724 | 0.57313 | 0.00306 | 0.00299 |
| ASMC s100 ct | docking_step | eskf1 | 0.29747 | 0.00857 | 0.56664 | 0.00306 | 0.00366 |
| ASMC s100 ct | docking_a | eskf2 | 0.30005 | 0.01048 | 0.59068 | 0.00349 | 0.00265 |
| ASMC s100 ct | docking_step | eskf2 | 0.30774 | 0.01099 | 0.58605 | 0.00348 | 0.00252 |
| ASMC s100 ct | docking_a | eskf3 | 0.29977 | 0.02592 | 0.55696 | 0.00337 | 0.00187 |
| ASMC s100 ct | docking_step | eskf3 | 0.29633 | 0.01687 | 0.54813 | 0.00337 | 0.00159 |
| ASMC s100 ct | docking_a | eskf4 | 0.30415 | 0.01683 | 0.58542 | 0.00351 | 0.00236 |
| ASMC s100 ct | docking_step | eskf4 | 0.31062 | 0.01641 | 0.58001 | 0.00353 | 0.00247 |
| ASMC s100 ct | docking_a | eskf5 | 0.29022 | 0.01585 | 0.55696 | 0.00339 | 0.00302 |
| ASMC s100 ct | docking_step | eskf5 | 0.29752 | 0.01528 | 0.55354 | 0.00337 | 0.00275 |
| PID-CT s100 ct | docking_a | eskf1 | 0.29298 | 0.01497 | 0.56292 | 0.00306 | 0.00351 |
| PID-CT s100 ct | docking_step | eskf1 | 0.29260 | 0.00882 | 0.55516 | 0.00306 | 0.00388 |
| PID-CT s100 ct | docking_a | eskf2 | 0.30077 | 0.01551 | 0.58097 | 0.00350 | 0.00289 |
| PID-CT s100 ct | docking_step | eskf2 | 0.30110 | 0.01099 | 0.57081 | 0.00348 | 0.00259 |
| PID-CT s100 ct | docking_a | eskf3 | 0.27762 | 0.00838 | 0.54578 | 0.00338 | 0.00154 |
| PID-CT s100 ct | docking_step | eskf3 | 0.28212 | 0.00765 | 0.53646 | 0.00338 | 0.00135 |
| PID-CT s100 ct | docking_a | eskf4 | 0.29482 | 0.00924 | 0.58252 | 0.00351 | 0.00317 |
| PID-CT s100 ct | docking_step | eskf4 | 0.30003 | 0.01192 | 0.56673 | 0.00353 | 0.00315 |
| PID-CT s100 ct | docking_a | eskf5 | 0.28230 | 0.01153 | 0.54847 | 0.00341 | 0.00346 |
| PID-CT s100 ct | docking_step | eskf5 | 0.28624 | 0.00861 | 0.54306 | 0.00340 | 0.00374 |

## Per-config mean (both trajectories, noise seeds averaged)

| config | traj | score | tracking | chatter | est_pos [m] | est_vel [m/s] |
|---|---|---|---|---|---|---|
| ASMC 0.9 ct | docking_a | 0.29832 | 0.01253 | 0.58098 | 0.00336 | 0.00228 |
| ASMC 0.9 ct | docking_step | 0.30415 | 0.01147 | 0.57535 | 0.00336 | 0.00234 |
| PID-CT 0.9 ct | docking_a | 0.28824 | 0.01013 | 0.56484 | 0.00337 | 0.00290 |
| PID-CT 0.9 ct | docking_step | 0.29169 | 0.00760 | 0.55711 | 0.00337 | 0.00284 |
| ASMC s100 ct | docking_a | 0.29679 | 0.01526 | 0.57263 | 0.00336 | 0.00258 |
| ASMC s100 ct | docking_step | 0.30194 | 0.01362 | 0.56687 | 0.00336 | 0.00260 |
| PID-CT s100 ct | docking_a | 0.28970 | 0.01192 | 0.56413 | 0.00337 | 0.00291 |
| PID-CT s100 ct | docking_step | 0.29242 | 0.00960 | 0.55444 | 0.00337 | 0.00294 |
