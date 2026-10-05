#!/bin/zsh
# HYSPLIT practice run: Prairie State Generating Station, OCO-2 overpass 2019-09-14 19:08 UTC.
# Emission 853,200 kg CO2/h = 237 kg/s (EPA CAMPD, 12-14 local). Stack 213 m. Met: NAM 12 km ARL file.
# Usage: ./run.sh /path/to/hysplit/exec
set -e
EXEC=${1:-$HOME/hysplit/exec}
cd "$(dirname "$0")"
echo "== trajectory test (hyts_std) =="; cp CONTROL.traj CONTROL; "$EXEC/hyts_std"; head -30 tdump
echo "== dispersion (hycs_std) =="; cp CONTROL.disp CONTROL; "$EXEC/hycs_std"
echo "== concentration grid -> ASCII (con2asc) =="; "$EXEC/con2asc" -icdump -m -z   # -m multiple files, -z zero-conc rows kept
ls -la cdump* tdump
