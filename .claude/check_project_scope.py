#!/usr/bin/env python3
"""
PreToolUse hook for Bash and PowerShell.
Triggers a permission prompt if a command references absolute paths
outside the mecanum_pinn_head tree, or writes to AppData/system dirs.
Relative paths (within the project) are always silently allowed.
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

# Windows absolute path: starts with a drive letter like C:\ or C:/
WIN_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]")

# Keywords that indicate a write/modify operation (not just a read)
WRITE_KEYWORDS = (
    ">", ">>", " cp ", " copy ", " mv ", " move ", " rm ", " del ",
    "mkdir", "New-Item", "Copy-Item", "Move-Item", "Remove-Item",
    "Out-File", "Set-Content", "Add-Content", "Rename-Item",
)

# System locations that are risky to write to
SYSTEM_PATH_RE = re.compile(
    r"AppData|%APPDATA%|\$env:APPDATA|%TEMP%|\$env:TEMP"
    r"|C:\\Windows|C:\\Program Files|C:\\ProgramData",
    re.IGNORECASE,
)


def is_outside_project(token: str) -> bool:
    if not WIN_ABS_RE.match(token):
        return False
    if PROJECT_ANCHOR in token:
        return False
    norm = token.replace("/", "\\").lower()
    if any(norm.startswith(prefix) for prefix in ALLOWED_EXTERNAL_PREFIXES):
        return False
    return True


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # can't parse → let permission system decide normally

    command = data.get("tool_input", {}).get("command", "")
    if not command:
        sys.exit(0)

    # Split on whitespace and common shell delimiters to get individual tokens
    tokens = re.split(r'[\s;|&"\'`()]+', command)

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
