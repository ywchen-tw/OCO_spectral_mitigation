"""Follow-up diagnostics for the weighting test (reads run_weighting.py outputs).

1. Systematic vs random misfit: per channel, split the far-clear first-pass
   residual mean square into the across-sounding mean² (a fixed channel
   population that repeats in every sounding) and the across-sounding
   variance (what can make k1 scatter).
2. Near-cloud residual structure in the channels Huber removes only near
   clouds (band-edge continuum).
3. Calibration of σ(k1): observed far-clear spread vs the formal noise σ and
   the calibrated, autocorrelation-inflated σ.
4. Latitude-local near-cloud contrast of k1 *and* k2, with bootstrap CIs, and
   a joint near/far separability (Fisher distance) on (k1, k2) per band and on
   all six coefficients, to see whether a weighting loses the near-cloud
   information or only moves it between k1 and k2.

Usage:
    python idea_test/fit_order_study/weight_extra.py --l1b ... --fp-tau ... --fit-details ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import FAR_KM, NEAR_KM, md_table  # noqa: E402
from run_study import robust_sd  # noqa: E402
from run_weighting import (LOCAL_REF_KM, MID_KM, first_pass_resid, lat_local_ref,  # noqa: E402
                          solar_line_depth, wavelengths)

KEY_TAGS = ("ols_2", "wls_2", "vf_2", "vfm_2", "vfh_2", "vfs_2", "vfc_2",
            "gamma_u", "gamma", "vf_g", "vfm_g", "vfs_g", "sm_2", "sm_g")
N_BOOT = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b", required=True)
    ap.add_argument("--fp-tau", required=True)
    ap.add_argument("--fit-details", required=True)
    ap.add_argument("--orbit", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    orbit = args.orbit or Path(args.fp_tau).parent.name
    out_dir = Path(args.out_dir)

    orb = load_orbit(args.l1b, args.fp_tau, args.fit_details)
    n = len(orb.sounding_id)
    cld, qf, fp = orb.meta["cld_dist_km"], orb.meta["xco2_qf"], orb.fp_number
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    tau, lnt = orb.tau[:, :, 1:-1], orb.ln_t[:, :, 1:-1]
    long = pd.read_parquet(out_dir / f"weight_per_sounding_{orbit}.parquet")
    wl = wavelengths(args.l1b)[:, :, 1:-1]

    rows, edge_rows = [], []
    for ib, band in enumerate(BANDS):
        d = long[(long.band == band) & (long.variant == "ols_2")].set_index("j").reindex(np.arange(n))
        params = d[["intercept", "k1", "k2"]].values
        r = first_pass_resid("c2", params, tau[ib], lnt[ib])
        sys_ms, rnd_ms, tot_ms = 0.0, 0.0, 0.0
        sys_lo, rnd_lo = 0.0, 0.0
        for ifp in range(8):
            rr = r[far & (fp == ifp)]
            mu = np.nanmean(rr, axis=0)
            var = np.nanvar(rr, axis=0)
            ok = np.isfinite(mu) & np.isfinite(var)
            sys_ms += np.sum(mu[ok] ** 2)
            rnd_ms += np.sum(var[ok])
            tm = np.nanmedian(tau[ib][far & (fp == ifp)], axis=0)
            lo = ok & (tm <= np.nanquantile(tm, 1 / 3))
            sys_lo += np.sum(mu[lo] ** 2)
            rnd_lo += np.sum(var[lo])
        rows.append(dict(band=band, systematic_share=sys_ms / (sys_ms + rnd_ms),
                         systematic_share_low_tau=sys_lo / (sys_lo + rnd_lo)))

        # band-edge continuum: first 60 channels (short-wavelength end), low τ
        tm_all = np.nanmedian(tau[ib][far], axis=0)
        for name, sl in (("short_edge", slice(0, 60)), ("long_edge", slice(-60, None))):
            ch = np.arange(tau.shape[2])[sl]
            ch = ch[tm_all[ch] <= np.nanquantile(tm_all, 0.25)]
            if len(ch) < 5:
                continue
            for cls_name, m in (("far", far), ("near", near)):
                rm = np.nanmean(r[m][:, ch], axis=1)
                edge_rows.append(dict(band=band, region=name, cls=cls_name, n_chan=len(ch),
                                      wl_nm=f"{wl[ib, 0, ch[0]] * 1e3:.1f}-{wl[ib, 0, ch[-1]] * 1e3:.1f}",
                                      mean_resid=np.nanmedian(rm), p10=np.nanpercentile(rm, 10),
                                      p90=np.nanpercentile(rm, 90)))
    sysrand = pd.DataFrame(rows)
    edge = pd.DataFrame(edge_rows)

    s = pd.read_csv(out_dir / f"weight_summary_{orbit}.csv")
    s = s[s.sig_k1_infl_pct.notna()].copy()
    s["spread_over_noise_sigma"] = s.k1_spread_pct / s.sig_k1_pct
    s["spread_over_infl_sigma"] = s.k1_spread_pct / s.sig_k1_infl_pct
    cal = s[["band", "variant", "k1_spread_pct", "sig_k1_pct", "sig_k1_cal_pct", "sig_k1_infl_pct",
             "spread_over_noise_sigma", "spread_over_infl_sigma"]]

    sep = separability(long, orb.meta, fp, np.random.default_rng(2))
    sol = solar_share(args, far, fp, out_dir, orbit)

    sep.to_csv(out_dir / f"weight_extra_local_signal_{orbit}.csv", index=False)
    sol.to_csv(out_dir / f"weight_extra_solar_{orbit}.csv", index=False)
    sysrand.to_csv(out_dir / f"weight_extra_sysrand_{orbit}.csv", index=False)
    edge.to_csv(out_dir / f"weight_extra_edge_{orbit}.csv", index=False)
    cal.to_csv(out_dir / f"weight_extra_sigma_calibration_{orbit}.csv", index=False)
    print(md_table(sysrand, list(sysrand.columns)))
    print()
    print(md_table(edge, list(edge.columns)))
    print()
    print(md_table(cal, list(cal.columns)))
    print()
    print(md_table(sep, list(sep.columns)))
    print()
    print(md_table(sol, list(sol.columns)))


def _clip_cov(x):
    """Covariance after clipping each column at ±5 robust SD about its median."""
    med = np.nanmedian(x, axis=0)
    sd = np.array([robust_sd(c) for c in x.T])
    xc = np.clip(x, med - 5 * sd, med + 5 * sd)
    ok = np.all(np.isfinite(xc), axis=1)
    return np.cov(xc[ok].T), sd


def separability(long, meta, fp, rng):
    """Latitude-local near/far contrast per coefficient and joint Fisher distance."""
    cld, qf, lat = meta["cld_dist_km"], meta["xco2_qf"], meta["lat"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    mid = (cld >= NEAR_KM) & (cld < MID_KM)
    ref = (cld > LOCAL_REF_KM) & (qf == 0)
    n = len(cld)
    rows = []
    # one set of draws shared by all variants, so variant differences are paired
    i_far, i_near_all = np.flatnonzero(far), np.flatnonzero(near)
    boots = [(rng.choice(i_far, len(i_far)), rng.choice(i_near_all, len(i_near_all)))
             for _ in range(N_BOOT)]
    for tag in KEY_TAGS:
        feats = {}
        for band in BANDS:
            d = long[(long.band == band) & (long.variant == tag)].set_index("j").reindex(np.arange(n))
            for coef in ("k1", "k2"):
                x = d[coef].values
                fpm = pd.Series(x[far]).groupby(fp[far]).mean()
                scale = np.nanmedian(x[far])
                rel = (x - fpm.reindex(fp).values) / scale          # relative to the far-clear median
                feats[(band, coef)] = rel - lat_local_ref(rel, lat, ref)
        names = list(feats)
        x_all = np.column_stack([feats[k] for k in names])
        cov_far, sd_far = _clip_cov(x_all[far])

        def fisher(idx, rows_n, rows_f, cov=None):
            dmu = np.nanmedian(x_all[rows_n][:, idx], axis=0) - np.nanmedian(x_all[rows_f][:, idx], axis=0)
            c = (cov_far if cov is None else cov)[np.ix_(idx, idx)]
            return float(np.sqrt(dmu @ np.linalg.solve(c, dmu)))

        def fisher_ci(idx):
            vals = [fisher(idx, b2, b1, _clip_cov(x_all[b1])[0]) for b1, b2 in boots[:100]]
            return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
        for ib, band in enumerate(BANDS):
            i1, i2 = names.index((band, "k1")), names.index((band, "k2"))
            for coef, ii in (("k1", i1), ("k2", i2)):
                xs = x_all[:, ii]
                snr = np.nanmedian(xs[near]) / sd_far[ii]
                bs = [np.nanmedian(xs[b2]) / robust_sd(xs[b1]) for b1, b2 in boots]
                rows.append(dict(variant=tag, band=band, feature=coef,
                                 sd_far_loc_pct=100 * sd_far[ii],
                                 n_near_loc=int(np.isfinite(xs[near]).sum()),
                                 shift_near_pct=100 * np.nanmedian(xs[near]),
                                 shift_mid_pct=100 * np.nanmedian(xs[mid]),
                                 snr_near_loc=snr, snr_lo=np.percentile(bs, 2.5),
                                 snr_hi=np.percentile(bs, 97.5),
                                 snr_mid_loc=np.nanmedian(xs[mid]) / sd_far[ii]))
            rows.append(dict(variant=tag, band=band, feature="fisher(k1,k2)",
                             snr_near_loc=fisher([i1, i2], near, far),
                             snr_mid_loc=fisher([i1, i2], mid, far)))
        all_idx = list(range(len(names)))
        k1_idx = [names.index((b, "k1")) for b in BANDS]
        for feat, idx in (("fisher(6 coef)", all_idx), ("fisher(3 k1)", k1_idx)):
            lo, hi = fisher_ci(idx)
            rows.append(dict(variant=tag, band="all", feature=feat,
                             snr_near_loc=fisher(idx, near, far), snr_lo=lo, snr_hi=hi,
                             snr_mid_loc=fisher(idx, mid, far)))
    return pd.DataFrame(rows)


def solar_share(args, far, fp, out_dir, orbit):
    """Share of Huber down-weight events in solar-line channels, by depth threshold."""
    z = np.load(out_dir / f"weight_downweight_freq_{orbit}.npz")
    classes = list(z["classes"])
    depth = z["solar_depth"].mean(axis=1)                      # [3, C], mean over fp
    rows = []
    for tag in ("vfh_2", "vfch_2"):
        freq = z[tag]                                           # [class, 3, C]
        for ib, band in enumerate(BANDS):
            for thr in (0.02, 0.05, 0.1):
                sel = depth[ib] > thr
                rec = dict(variant=tag, band=band, depth_threshold=thr,
                           frac_channels=float(sel.mean()))
                for c in ("far", "near"):
                    f = freq[classes.index(c), ib]
                    rec[f"share_events_{c}"] = float(f[sel].sum() / f.sum()) if f.sum() > 0 else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
