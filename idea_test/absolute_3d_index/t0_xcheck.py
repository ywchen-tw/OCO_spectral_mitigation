"""T0: verify the Rayleigh floor geometry X before any X-based quantity is used.

Manuscript SI S1.6 (CURC, 6 dates 2016-2020, QF 0, cloud distance > 10 km):
across cos(SZA) quintiles the offset form f*A*mu0 spreads ~50 % (52 land /
47 water) against 16 % for f*A/X; f*A/X = 0.22 land / 0.11 water;
per-footprint IQR/median 33 % (offset) vs 16 % (Rayleigh); obs-mode spread
over land 33 % vs 7 %.

Spread = (max - min) / median of the per-quintile medians (the PLAN.md wording
"changes by" is read this way; the alternative (max - min)/mean is printed too).
Both azimuth sign conventions are evaluated: the one used here (glint at
raa = 180, cos Theta = -mu0 mu - s0 s cos raa) and the opposite sign used by the
stored cos_theta column.  Population: the 139 evaluation dates, QF 0, snow-free,
cloud distance > 10 km.

Output: out/t0_xcheck.csv (+ stdout)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common


def spread(v: pd.Series, key: pd.Series) -> tuple[float, float]:
    med = v.groupby(key, observed=True).median()
    return (med.max() - med.min()) / med.median(), (med.max() - med.min()) / med.mean()


def main() -> None:
    cols = ["sfc_type", "xco2_qf", "cld_dist_km", "sza", "vza", "o2a_f_am", "alb_o2a",
            "X", "X_wrongsign", "operation_mode", "glint_angle", "raa", "date"]
    df = common.load_table(cols)
    df = df[(df.xco2_qf == 0) & (df.cld_dist_km > 10)].copy()
    mu0 = np.cos(np.radians(df.sza))
    df["offset"] = df.o2a_f_am * df.alb_o2a * mu0
    df["ray"] = df.o2a_f_am * df.alb_o2a / df.X
    df["ray_wrong"] = df.o2a_f_am * df.alb_o2a / df.X_wrongsign
    rows = []
    for sname, code in (("land", 1), ("water", 0)):
        s = df[df.sfc_type == code]
        s = s[np.isfinite(s.ray) & (s.ray > 0) & np.isfinite(s.offset)]
        q = pd.qcut(np.cos(np.radians(s.sza)), 5, labels=False)
        print(f"\n{sname}: n = {len(s):,}; obs modes {s.operation_mode.value_counts().to_dict()}")
        for name in ("offset", "ray", "ray_wrong"):
            v = s[name]
            sp_med, sp_mean = spread(v, q)
            iqr = (v.quantile(0.75) - v.quantile(0.25)) / v.median()
            sp_mode = spread(v, s.operation_mode)[0] if s.operation_mode.nunique() > 1 else np.nan
            rec = dict(surface=sname, form=name, n=len(v), median=v.median(),
                       mu0_quintile_spread=sp_med, mu0_quintile_spread_vs_mean=sp_mean,
                       iqr_over_median=iqr, obs_mode_spread=sp_mode,
                       quintile_medians=" ".join(f"{m:.4g}" for m in
                                                 v.groupby(q).median().to_numpy()))
            rows.append(rec)
            print(f"  {name:10s} median {v.median():.4g}  mu0-quintile spread "
                  f"{100 * sp_med:5.1f} %  IQR/med {100 * iqr:5.1f} %  mode spread "
                  f"{100 * sp_mode:5.1f} %  quintile medians {rec['quintile_medians']}")
    pd.DataFrame(rows).to_csv(common.OUT / "t0_xcheck.csv", index=False)


if __name__ == "__main__":
    main()
