# Tier B — open-loop biased odd-multisine BLA

f0=0.1 Hz, excited bins k≡1 mod 4 up to 10.0 Hz, 6 periods (drop 2), Fs=500.0 Hz, per-period detrended. Bias = task-space voltage holding the cruising operating point (Tier A calibration); excitation = multisine RMS voltage around it. No measurement noise: period variance = stochastic nonlinear distortion.

| axis | bias [V] | exc [V] | v_mean/cap | odd NL [dB] | even NL [dB] | total NL [dB] | stochastic [dB] |
|---|---|---|---|---|---|---|---|
| 1 | 2.64 | 2.64 | 0.06 | -20.5 | -14.5 | -13.5 | -64.9 |
| 1 | 2.64 | 4.09 | 0.07 | -20.9 | -14.5 | -13.6 | -60.1 |
| 2 | 2.85 | 2.85 | 0.05 | -23.4 | -15.2 | -14.6 | -52.1 |
| 2 | 2.85 | 4.78 | 0.06 | -24.8 | -16.6 | -16.0 | -51.1 |
| 3 | 4.84 | 4.84 | 0.05 | -29.1 | -20.1 | -19.6 | -48.5 |
| 3 | 4.84 | 8.63 | 0.05 | -23.6 | -22.4 | -19.9 | -49.3 |

(dB = 10*log10(distortion power / excited-bin power); more negative = more linear.)

## BLA gain (driven channel)

| axis | bias [V] | exc [V] | band-median \|G\| | \|G(0.5 Hz)\| |
|---|---|---|---|---|
| 1 | 2.64 | 2.64 | 0.0097 | 0.0179 |
| 1 | 2.64 | 4.09 | 0.0090 | 0.0164 |
| 2 | 2.85 | 2.85 | 0.0084 | 0.0139 |
| 2 | 2.85 | 4.78 | 0.0080 | 0.0129 |
| 3 | 4.84 | 4.84 | 0.0269 | 0.0454 |
| 3 | 4.84 | 8.63 | 0.0235 | 0.0477 |
