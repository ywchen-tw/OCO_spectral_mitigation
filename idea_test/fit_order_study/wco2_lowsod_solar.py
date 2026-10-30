"""Per-footprint ILS-convolved solar spectrum on a grid of Doppler velocities (WCO2 low-SOD study, 2026-10-30).

ln S_conv(v) [3 bands, 8 fp, V_GRID, 1016] with the solar line list at λ_rest·(1 + v/c), production
convolution (fp_abs_coeff.py: solar model absorption × interpolated continuum, ILS-weighted mean).
Channel-independent factors (1/d², 1/cos sza, 1/(1+β)) are left out; they only move the fit intercept.
Used to (i) reproduce the production TOA solar (v = v_sol) up to a per-sounding constant and (ii) test
the solar Doppler hypothesis (v = v_sol ± v_inst).

Usage:  python wco2_lowsod_solar.py [GL|KOZ]
Output: results/wco2_lowsod_solar_grid_<tag>.npz (lnS, v_grid)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402

L1B = {"GL": wc.fm.DEFAULTS["l1b"],
       "KOZ": wc.ROOT / "data/OCO2/2021/249/38200a_ND/oco2_L1bScND_38200a_210906_B11006r_220325042044.h5"}


def _one(args):
    tag, ib, fp = args
    ils = wc.ils_tables(L1B[tag])[ib]
    sol = wc.solar_tables()[ib]
    out = np.empty((len(wc.V_GRID), 1016))
    for iv, v in enumerate(wc.V_GRID):
        out[iv] = np.log(wc._conv_solar(sol["wvl"] * (1 + v / wc.C_LIGHT), sol["fsol"],
                                        ils["wl"][fp], ils["xx"][fp], ils["yy"][fp]))
    return ib, fp, out


def main(tag):
    t0 = time.time()
    jobs = [(tag, ib, fp) for ib in range(3) for fp in range(8)]
    lnS = np.full((3, 8, len(wc.V_GRID), 1016), np.nan)
    with mp.get_context("fork").Pool(8) as pool:
        for ib, fp, out in pool.imap_unordered(_one, jobs):
            lnS[ib, fp] = out
    np.savez_compressed(wc.OUT / f"wco2_lowsod_solar_grid_{tag}.npz", lnS=lnS.astype(np.float32), v_grid=wc.V_GRID)
    print(f"{tag}: solar grid in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GL")
