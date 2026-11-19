#!/usr/bin/env bash
# Compile nd_analysis.tex with the folder-local tectonic (cache kept in tools/cache).
set -e
cd "$(dirname "$0")"
XDG_CACHE_HOME="$PWD/tools/cache" ./tools/tectonic "${1:-nd_analysis.tex}"
