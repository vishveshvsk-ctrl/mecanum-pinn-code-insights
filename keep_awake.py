#!/usr/bin/env python
# =============================================================================
# keep_awake.py — block Windows Modern Standby for the duration of a long run.
#
# Modern Standby suspends background processes ~30 min into idle even on AC,
# which silently kills long Julia sweeps / diagnostic re-runs. This daemon holds
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED via SetThreadExecutionState. The flag is
# THREAD-scoped, so this process must stay alive (run it in the background) for
# the whole run; when it exits, the keep-awake lapses.
#
# Launch it in the background BEFORE kicking off a long task, e.g.:
#   <venv python> keep_awake.py        (run_in_background)
# then start the sweep/re-run. Kill it (or let it be killed) when done.
# =============================================================================
import ctypes
import datetime
import time

ES_CONTINUOUS        = 0x80000000
ES_SYSTEM_REQUIRED   = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040   # away-mode: keep working with the lid logic engaged
FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED

set_state = ctypes.windll.kernel32.SetThreadExecutionState
r = set_state(FLAGS)
ts = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
print(f'keep-awake armed: {ts}  SetThreadExecutionState returned {r} (nonzero = ok)', flush=True)

try:
    # Re-assert every 60 s — a single call sets the state, but periodic re-arming
    # is cheap insurance against any other process clearing ES_CONTINUOUS.
    while True:
        set_state(FLAGS)
        time.sleep(60)
except KeyboardInterrupt:
    set_state(ES_CONTINUOUS)   # clear the requirement, allow normal sleep again
    print('keep-awake released', flush=True)
