"""Sign and slope of the telluric Doppler correction on target-mode orbit 48443a_TG.

On 29252a_GL the far-clear soundings sat in two v_inst clusters collinear with
latitude. Here ``sounding_relative_velocity`` sweeps −5.6…+5.8 km/s inside one
target scene. For ~600 QF0 soundings stratified in signed v (12 bins, all fp):

1. τ on the production grid (no shift; also a reproduction check), the corrected
   grid λ_ABSCO·(1 − v/c) (sign found on 29252a_GL) and the wrong sign λ·(1 + v/c);
2. a scan of the ABSCO-grid shift ε (λ·(1 + ε)), expressed in pixels
   δ = ε·λ̄/Δλ̄, giving each sounding's best shift by parabolic refinement of the
   order-3 residual rms (no linearization). The physical expectation for the
   corrected sign is δ = −(λ̄/(cΔλ̄))·v ≈ −0.174 px per km/s;
3. the linear estimator used on 29252a_GL (β on ∂τ/∂pix, δ_lin = −β/k1) on
   production τ, to show how much of any slope deficit is estimator attenuation;
4. order-3 / order-2 k1, k2, k2 zero fraction and the k1-vs-VZA slope, before and
   after the correction.

Usage:
    python idea_test/fit_order_study/resid_doppler_tg.py [--n-per-bin 50] [--workers 8]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from resid_fitters import fit_one  # noqa: E402
from run_study import md_table  # noqa: E402

ORBIT = "48443a_TG"
DATA = ROOT / "data/OCO2/2023/222/48443a_TG"
ra.L1B = DATA / "oco2_L1bScTG_48443a_230810_B11008r_230912173118.h5"
ra.MET = DATA / "oco2_L2MetTG_48443a_230810_B11008r_230912180307.h5"
ra.CPR = DATA / "oco2_L2CPrTG_48443a_230810_B11008r_230912235548.h5"
ra.FP_TAU = HERE / "results/pipeline_2023-08-10/2023-08-10/48443a_TG/fp_tau_combined.h5"
ra.FIT_DETAILS = HERE / "results/pipeline_2023-08-10/fitting_details/fitting_details_2023-08-10_48443a_TG.h5"
OUT = HERE / "results"
PREFIX = "resid_doppler_tg"
C_LIGHT = 299792458.0
N_BINS = 12
SCAN_PX = np.round(np.arange(-1.3, 1.31, 0.1), 2)     # ABSCO-grid shift scan (pixels)
_G = ra._G


def _sounding(i: int) -> dict:
    """All τ variants for sounding position i (3 bands)."""
    atm, bands = _G["atm"], _G["bands"]
    fp = int(_G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(_G["sza"][i])) + 1 / np.cos(np.deg2rad(_G["vza"][i]))
    beta = atm["v_inst"][i] / C_LIGHT
    dz = atm["dz"][i]
    out = {}
    for ib, b in enumerate(bands):
        gas_den = atm["o2"][i] if ib == 0 else atm["co2"][i]
        e_gas, e_h2o = ra._ext_layers(b, atm, i, gas_den)
        ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
        tau_hr = ((e_gas + e_h2o) * dz).sum(axis=1) * airmass + ray * airmass
        wl, xx, yy, wv = b["wl"][fp], b["xx"][fp], b["yy"][fp], b["wvldat"]
        px_frac = np.mean(np.gradient(wl) / wl)                 # Δλ/λ per pixel (band mean)
        conv = lambda eps, wl_=wl: ra._tau_conv(tau_hr, wv * (1.0 + eps), wl_, xx, yy)  # noqa: E731
        dwl = np.gradient(wl)
        out[BANDS[ib]] = dict(
            prod=conv(0.0), corr=conv(-beta), wrong=conv(beta),
            scan=np.stack([conv(d * px_frac) for d in SCAN_PX]),
            dtau_dpix=(ra._tau_conv(tau_hr, wv, wl + 0.1 * dwl, xx, yy)
                       - ra._tau_conv(tau_hr, wv, wl - 0.1 * dwl, xx, yy)) / 0.2,
            px_per_kms=1e3 / C_LIGHT / px_frac)
    return out


def _work(idx):
    return [(i, _sounding(i)) for i in idx]


def select(v, fp, qf, n_per_bin, rng):
    """n_per_bin QF0 soundings per signed-v bin, round-robin over footprints."""
    ok = np.flatnonzero((qf == 0) & np.isfinite(v))
    edges = np.linspace(v[ok].min(), v[ok].max() + 1e-6, N_BINS + 1)
    sel = []
    for k in range(N_BINS):
        cand = ok[(v[ok] >= edges[k]) & (v[ok] < edges[k + 1])]
        by_fp = [rng.permutation(cand[fp[cand] == f]).tolist() for f in range(8)]
        pick = []
        while len(pick) < n_per_bin and any(by_fp):
            for lst in by_fp:
                if lst and len(pick) < n_per_bin:
                    pick.append(lst.pop())
        sel += pick
    return np.array(sorted(sel)), edges


def _parabola_min(x, y):
    k = int(np.nanargmin(y))
    if k == 0 or k == len(x) - 1:
        return x[k]
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    den = y0 - 2 * y1 + y2
    return x[k] + (0.5 * (y0 - y2) / den * (x[1] - x[0]) if den > 0 else 0.0)


def vza_slope(k1, vza):
    """Pooled slope of k1/median(k1) per 10° VZA (%)."""
    ok = np.isfinite(k1) & np.isfinite(vza)
    return 100 * 10 * np.polyfit(vza[ok], k1[ok] / np.median(k1[ok]), 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bin", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    orb = load_orbit(str(ra.L1B), str(ra.FP_TAU), str(ra.FIT_DETAILS))
    with h5py.File(ra.FIT_DETAILS, "r") as f:
        vza_all = f["vza"][...]
    with h5py.File(ra.L1B, "r") as f:
        lsid = f["SoundingGeometry/sounding_id"][...].ravel()
        lv = f["SoundingGeometry/sounding_relative_velocity"][...].ravel() / 1e3
    pos = {s: i for i, s in enumerate(lsid)}
    v_all = np.array([lv[pos[s]] for s in orb.sounding_id])
    sel, edges = select(v_all, orb.fp_number, orb.meta["xco2_qf"], args.n_per_bin, np.random.default_rng(0))
    print(f"{len(sel)} soundings, v {v_all[sel].min():.2f}…{v_all[sel].max():.2f} km/s", flush=True)

    with h5py.File(ra.FP_TAU, "r") as f:
        sza, vza_tau = f["sza"][...], f["vza"][...]
    bands, h2o_meta = ra.load_tables()
    atm = ra.build_atmosphere(orb.sounding_id[sel])
    _G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=orb.fp_number[sel], sza=sza[sel], vza=vza_tau[sel])
    t0 = time.time()
    chunks = np.array_split(np.arange(len(sel)), max(1, len(sel) // 4))
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = sorted([r for part in pool.imap(_work, chunks) for r in part], key=lambda x: x[0])
    print(f"τ variants for {len(sel)} soundings in {time.time() - t0:.0f} s", flush=True)

    rows, repro = [], {}
    for ib, band in enumerate(BANDS):
        rel = []
        for k, j in enumerate(sel):
            r = res[k][1][band]
            ref = orb.tau[ib, j].astype(float)
            rel.append(np.nanmedian(np.abs(r["prod"] - ref) / ref))
            y_all = orb.ln_t[ib, j, 1:-1].astype(float)
            rec = dict(j=j, band=band, fp=orb.fp_number[j], v=v_all[j], vza=vza_all[j],
                       px_per_kms=r["px_per_kms"])
            for name in ("stored", "prod", "corr", "wrong"):
                t = (ref if name == "stored" else r[name])[1:-1]
                m = np.isfinite(t) & np.isfinite(y_all)
                for order in (3, 2):
                    f = fit_one(t[m], y_all[m], order, {})
                    rec[f"rms{order}_{name}"] = f["resid_rms"]
                    rec[f"k1_{order}_{name}"] = f["k1"]
                    rec[f"k2_{order}_{name}"] = f["k2"]
            t = r["prod"][1:-1]
            m = np.isfinite(t) & np.isfinite(y_all)
            scan = [fit_one(s[1:-1][m], y_all[m], 3, {})["resid_rms"] for s in r["scan"]]
            rec["shift_best_px"] = _parabola_min(SCAN_PX, np.array(scan))
            rec["shift_best_at_edge"] = bool(np.argmin(scan) in (0, len(SCAN_PX) - 1))
            g = r["dtau_dpix"][1:-1][m]
            f = fit_one(t[m], y_all[m], 3, {"disp": g})
            rec["shift_lin_px"] = -f["x_disp"] / f["k1"]           # channel-grid shift (attrs sign)
            rows.append(rec)
        repro[band] = (float(np.median(rel)), float(np.percentile(rel, 99)))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"{PREFIX}_per_sounding_{ORBIT}.parquet", index=False)
    write_report(df, edges, repro, orb, v_all, vza_all)


def write_report(df, edges, repro, orb, v_all, vza_all):
    df = df.copy()
    df["vbin"] = np.clip(np.searchsorted(edges, df.v, side="right") - 1, 0, N_BINS - 1)
    parts = [f"# Doppler sign/slope test — target orbit {ORBIT}\n",
             f"{df.j.nunique()} QF0 soundings, {N_BINS} bins of signed v (sounding_relative_velocity), "
             "all fp. corr: ABSCO grid λ(1 − v/c) (29252a_GL sign); wrong: λ(1 + v/c).\n",
             "Reproduction of stored τ by the unshifted recomputation (median per-sounding median "
             "|Δτ|/τ, p99): " + ", ".join(f"{b} {m:.1e} ({p:.1e})" for b, (m, p) in repro.items()) + "\n"]

    # 1. rms per band × v bin
    g = df.groupby(["band", "vbin"])
    t = g.agg(v_lo=("v", "min"), v_hi=("v", "max"), n=("j", "size"),
              rms_prod=("rms3_stored", "median"), rms_corr=("rms3_corr", "median"),
              rms_wrong=("rms3_wrong", "median"),
              frac_corr_better=("rms3_corr", lambda s: np.nan)).reset_index()
    t["frac_corr_better"] = [float(np.mean(d.rms3_corr < d.rms3_stored)) for _, d in g]
    t["frac_wrong_worse"] = [float(np.mean(d.rms3_wrong > d.rms3_stored)) for _, d in g]
    t.to_csv(OUT / f"{PREFIX}_rms_by_v_{ORBIT}.csv", index=False)
    parts.append("\n## Order-3 residual rms (median) per signed-v bin\n")
    parts.append(md_table(t, list(t.columns)))
    verdict = []
    for band in BANDS:
        s = t[t.band == band]
        pos, neg = s[s.v_lo > 0.5], s[s.v_hi < -0.5]
        ok = ((pos.rms_corr < pos.rms_prod).all() and (neg.rms_corr < neg.rms_prod).all()
              and (pos.rms_wrong > pos.rms_prod).all() and (neg.rms_wrong > neg.rms_prod).all())
        verdict.append(f"{band}: {'CONFIRMED' if ok else 'NOT confirmed'} "
                       f"(bins |v| > 0.5 km/s: {len(pos)} positive, {len(neg)} negative)")
    parts.append("\nSign test (corrected < production < wrong in every bin with |v| > 0.5 km/s): "
                 + "; ".join(verdict))

    # 2. implied shift vs v
    recs = []
    for band in BANDS:
        d = df[(df.band == band) & ~df.shift_best_at_edge]
        exp = float(d.px_per_kms.median())
        sl, ic = np.polyfit(d.v, d.shift_best_px, 1)
        resid = d.shift_best_px - (sl * d.v + ic)
        sl0 = float(np.sum(d.v * d.shift_best_px) / np.sum(d.v**2))   # through the origin
        dl = df[df.band == band]
        sll, icl = np.polyfit(dl.v, dl.shift_lin_px, 1)
        # free scale factor vs physical: rms of corrected-τ order-3 fit is the physical case;
        # the best-shift rms is ≤ the free-scale rms, so compare at the sounding level
        recs.append(dict(band=band, n=len(d), n_at_edge=int(df[df.band == band].shift_best_at_edge.sum()),
                         expected_px_per_kms=-exp, scan_slope=sl, scan_intercept_px=ic,
                         scan_slope_origin=sl0, scan_scale=sl / -exp, scan_resid_sd_px=float(resid.std()),
                         corr_shift_v=float(np.corrcoef(d.v, d.shift_best_px)[0, 1]),
                         lin_slope_channel_px=sll, lin_intercept_px=icl, lin_scale=sll / exp))
    sh = pd.DataFrame(recs)
    sh.to_csv(OUT / f"{PREFIX}_shift_{ORBIT}.csv", index=False)
    parts.append("\n## Implied shift vs signed v\n")
    parts.append("scan_*: best ABSCO-grid shift (px) from the rms scan (−1.3…+1.3 px, parabolic refinement; "
                 "soundings with the minimum at the scan edge excluded). Physical expectation for the "
                 "corrected sign: −λ̄/(cΔλ̄) px per km/s (`expected_px_per_kms`). lin_*: linear ∂τ/∂pix "
                 "estimator on production τ (channel-grid shift, opposite sign convention), as on 29252a_GL.\n")
    parts.append(md_table(sh, list(sh.columns)))

    # free scale factor: rms of best-shift vs physical correction
    fs = []
    for band in BANDS:
        d = df[df.band == band]
        fs.append(dict(band=band, rms3_prod=d.rms3_stored.median(), rms3_physical=d.rms3_corr.median(),
                       rms3_wrong=d.rms3_wrong.median()))
    parts.append("\n## Fit quality summary (median order-3 rms over all selected soundings)\n")
    parts.append(md_table(pd.DataFrame(fs), ["band", "rms3_prod", "rms3_physical", "rms3_wrong"]))

    # 3. k1/k2 before/after and VZA slope
    kr = []
    for band in BANDS:
        d = df[df.band == band]
        for order in (3, 2):
            for name, lab in (("stored", "production"), ("corr", "corrected")):
                k1, k2 = d[f"k1_{order}_{name}"].values, d[f"k2_{order}_{name}"].values
                kr.append(dict(band=band, order=order, tau=lab, k1_median=np.median(k1),
                               k2_median=np.median(k2), k2_zero_frac=float(np.mean(k2 <= 1e-6)),
                               k1_vza_slope_pct_per_10deg=vza_slope(k1, d.vza.values),
                               k1_v_slope_pct_per_kms=100 * np.polyfit(d.v, k1 / np.median(k1), 1)[0]))
    kt = pd.DataFrame(kr)
    kt.to_csv(OUT / f"{PREFIX}_k_{ORBIT}.csv", index=False)
    parts.append("\n## k1, k2 and k1-vs-VZA slope, before/after (selected soundings)\n")
    parts.append(md_table(kt, list(kt.columns)))
    # reference: production ols_3 VZA slope on all QF0 from the other agent's run
    try:
        p = pd.read_parquet(HERE / "results/tg_run/per_sounding_48443a_TG.parquet")
        qf0 = np.flatnonzero(orb.meta["xco2_qf"] == 0)
        ref = []
        for band in BANDS:
            for order in (3, 2):
                d = p[(p.band == band) & (p.variant == f"ols_{order}")].set_index("j").reindex(qf0)
                ref.append(f"{band} o{order} {vza_slope(d.k1.values, vza_all[qf0]):+.2f}")
        parts.append("\nReference, production τ on all QF0 soundings (tg_run parquet), k1-vs-VZA slope "
                     "% per 10°: " + ", ".join(ref))
    except (FileNotFoundError, KeyError) as exc:
        parts.append(f"\nReference slope not computed: {exc}")
    text = "\n".join(parts) + "\n"
    (OUT / f"{PREFIX}_summary_{ORBIT}.md").write_text(text)
    print(text)
    _figure(df, sh)


def _figure(df, sh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ib, band in enumerate(BANDS):
        d = df[df.band == band]
        ax = axes[0, ib]
        for col, c, lab in (("rms3_stored", "k", "production"), ("rms3_corr", "C2", "corrected λ(1−v/c)"),
                            ("rms3_wrong", "C3", "wrong sign λ(1+v/c)")):
            ax.scatter(d.v, d[col], s=3, color=c, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("sounding_relative_velocity (km/s)")
        ax.set_ylabel("order-3 residual rms")
        ax.set_title(band)
        ax = axes[1, ib]
        e = d[~d.shift_best_at_edge]
        ax.scatter(e.v, e.shift_best_px, s=3, c=e.fp, cmap="tab10")
        s = sh[sh.band == band].iloc[0]
        vv = np.linspace(-6, 6, 10)
        ax.plot(vv, s.expected_px_per_kms * vv, "k--", label=f"physical {s.expected_px_per_kms:.3f} px/(km/s)")
        ax.plot(vv, s.scan_slope * vv + s.scan_intercept_px, "C1-",
                label=f"fit {s.scan_slope:.3f}·v {s.scan_intercept_px:+.3f}")
        ax.set_xlabel("sounding_relative_velocity (km/s)")
        ax.set_ylabel("best ABSCO-grid shift (px)")
        ax.legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{PREFIX}_{ORBIT}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
