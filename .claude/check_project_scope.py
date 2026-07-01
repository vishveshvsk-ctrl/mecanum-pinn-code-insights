#!/usr/bin/env python3
"""
PreToolUse hook for Bash and PowerShell.
Triggers a permission prompt if a command references absolute paths
outside the mecanum_pinn_head tree, or writes to AppData/system dirs.
Relative paths (within the project) are always silently allowed.

Covers BOTH path dialects, since commands may run natively on Windows OR
be dispatched into WSL via `wsl.exe -e bash -lc '...'`:
  - Windows absolute paths   :  C:\\... / C:/...
  - WSL / POSIX paths         :  /mnt/c/..., /home/..., ~/..., $HOME/...
A `/mnt/<drive>/...` path is normalized back to its `<drive>:\\...` form so the
same project-anchor + allowlist logic applies regardless of how it was written.
The tokenizer splits quoted inner scripts apart, so paths inside a
`wsl ... bash -lc '...'` wrapper are checked too without a separate unwrap pass.
"""
import json
import re
import sys

PROJECT_ANCHOR = "mecanum_pinn_head"

# Trusted external trees: absolute paths under these prefixes are silently
# allowed even though they live outside the project. See memory
# project_python_env_tooling. Compared case-insensitively with both slash kinds.
#   - claude-venv/mecanum : dedicated Claude venv (matplotlib, pyarrow, scipy,
#     tectonic, nbconvert).
#   - miniforge3/envs/myenv : conda env holding fitz/PyMuPDF (PDF rasterization)
#     and the julia-1.12 IJulia kernel. Paths here have no spaces, so the token
#     splitter keeps them intact and the prefix match works.
ALLOWED_EXTERNAL_PREFIXES = (
    r"c:\users\vishv\claude-venv\mecanum",
    r"c:\users\vishv\miniforge3\envs\myenv",
)

# Ephemeral/virtual POSIX trees that are safe to read or write and would
# otherwise be noisy (e.g. `2>/dev/null`). Matched with a path-boundary so
# `/devices` does not get swallowed by `/dev`.
ALLOWED_POSIX_PREFIXES = ("/dev", "/proc", "/sys")

# Windows absolute path: starts with a drive letter like C:\ or C:/
WIN_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]")

# POSIX absolute path, ~ home expansion, or $HOME / ${HOME} rooted
POSIX_ABS_RE = re.compile(r"^(?:/|~/|~$|\$HOME\b|\$\{HOME\})")

# WSL drive mount:  /mnt/c/Users/...  ->  drive letter + remainder
MNT_RE = re.compile(r"^/mnt/([A-Za-z])/(.*)$")

# Keywords that indicate a write/modify operation (not just a read)
WRITE_KEYWORDS = (
    ">", ">>", " cp ", " copy ", " mv ", " move ", " rm ", " del ",
    " touch ", " tee ", " dd ", "truncate",
    "mkdir", "New-Item", "Copy-Item", "Move-Item", "Remove-Item",
    "Out-File", "Set-Content", "Add-Content", "Rename-Item",
)

# System locations that are risky to write to
SYSTEM_PATH_RE = re.compile(
    r"AppData|%APPDATA%|\$env:APPDATA|%TEMP%|\$env:TEMP"
    r"|C:\\Windows|C:\\Program Files|C:\\ProgramData",
    re.IGNORECASE,
)


def _win_outside_project(win_token: str) -> bool:
    """True if a Windows-style absolute path lives outside the project tree."""
    if PROJECT_ANCHOR in win_token:
        return False
    norm = win_token.replace("/", "\\").lower()
    if any(norm.startswith(prefix) for prefix in ALLOWED_EXTERNAL_PREFIXES):
        return False
    return True


def _posix_allowed(token: str) -> bool:
    """True for benign virtual POSIX trees (/dev, /proc, /sys)."""
    return any(
        token == p or token.startswith(p + "/") for p in ALLOWED_POSIX_PREFIXES
    )


def is_outside_project(token: str) -> bool:
    # 1. Native Windows absolute path (C:\... or C:/...)
    if WIN_ABS_RE.match(token):
        return _win_outside_project(token)

    # 2. WSL drive mount (/mnt/c/...) -> normalize to C:\... and reuse #1's logic
    mnt = MNT_RE.match(token)
    if mnt:
        drive, rest = mnt.group(1), mnt.group(2)
        win_rest = rest.replace("/", "\\")
        return _win_outside_project(drive + ":\\" + win_rest)

    # 3. Other POSIX / ~ / $HOME absolute path -> outside the Windows project tree
    if POSIX_ABS_RE.match(token):
        if _posix_allowed(token):
            return False
        if PROJECT_ANCHOR in token:
            return False
        return True

    # 4. Relative path or non-path token -> allowed
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # can't parse → let permission system decide normally

    command = data.get("tool_input", {}).get("command", "")
    if not command:
        sys.exit(0)

    # Split on whitespace, shell delimiters, AND redirect/assignment operators
    # (< > =) so that targets in `>/mnt/c/x`, `2>/dev/null`, and `--cwd=/path`
    # surface as bare path tokens.
    tokens = re.split(r'[\s;|&"\'`()<>=]+', command)

    # Check 1: any token is an absolute path outside the project
    risky_paths = [t for t in tokens if t and is_outside_project(t)]

    # Check 2: command writes to AppData / system dirs
    has_write_op = any(kw in command for kw in WRITE_KEYWORDS)
    risky_system = has_write_op and bool(SYSTEM_PATH_RE.search(command))

    if risky_paths or risky_system:
        reason = "references path(s) outside mecanum_pinn_head"
        if risky_paths:
            reason += f": {risky_paths[:3]}"
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": reason,
            }
        }))

    # No output → use normal permission handling (allow list applies)
    sys.exit(0)


if __name__ == "__main__":
    main()
