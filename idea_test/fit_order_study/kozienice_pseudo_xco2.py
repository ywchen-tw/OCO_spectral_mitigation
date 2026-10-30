"""Pseudo-XCO2 from the agreed fit model across the Kozienice plume (orbit 38200a_ND, 2021-09-06).

Known-answer test for the direct-XCO2 idea on land with real XCO2 variation: the L2 product shows a
plume at the Kozienice power plant (51.665 N, 21.467 E).  Everything is computed locally for the
soundings within RADIUS_KM of the plant, with the study's copy of the τ step:

  τ        : Doppler-corrected (sign −1), split into dry / H2O / Rayleigh (h2o_split.split_one)
  TOA solar: production formula (fp_abs_coeff: solar Doppler, 1/d², ILS convolution, × 1/cos sza)
  ln T     : ln(rad / (0.5 · toa) · π) as orbit_io.load_orbit
  fit      : agreed model (final_model_test.fit_o2a / fit_wco2 / fit_sco2), land from the L1B land fraction

Estimators are unscaled (τ in the CO2 bands is built with the prior CO2 profile):
  E0 = prior · k1_S, E1 = prior · k1_S / k1_O2A, E1w = prior · k1_W / k1_O2A,
  plus a plain single-τ linear fit (ln T = c − k1 τ, no continuum/H2O/floor terms) on the same corrected τ
  as the "before" reference.  Targets: Lite xco2 (bias-corrected), xco2_raw, xco2_strong_idp, the prior.
No retrieved quantity enters any fit; fs_rel is read for validation only.

Outputs: results/kozienice_pseudo_xco2.{md,csv,png}, results/kozienice_per_sounding.parquet.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import resid_attrs as ra  # noqa: E402
import final_model_test as fm  # noqa: E402
from h2o_split import split_one  # noqa: E402
from orbit_io import BANDS, _L1B_RAD, _M1, l1b_noise  # noqa: E402

OUT = HERE / "results"
DATA = ROOT / "data/OCO2/2021/249"
ORB = DATA / "38200a_ND"
FILES = dict(
    l1b=ORB / "oco2_L1bScND_38200a_210906_B11006r_220325042044.h5",
    met=ORB / "oco2_L2MetND_38200a_210906_B11006r_220322112311.h5",
    cpr=ORB / "oco2_L2CPrND_38200a_210906_B11006r_220322112932.h5",
    lite=DATA / "oco2_LtCO2_210906_B11210Ar_240916224317s.nc4",
    parquet=ROOT / "results/csv_collection/combined_2021-09-06_all_orbits.parquet",
    solar=ra.ABSCO_DIR / "l2_solar_model.h5",
)
PLANT = (51.66472, 21.46667)
RADIUS_KM = 150.0
CORE_KM, BG_KM = 10.0, 40.0
DOPPLER_SIGN = -1
C_LIGHT = 299792458.0
AU_M = 1.495978707e11
TAG = "kozienice_38200a_ND"


def km_from_plant(lat, lon):
    return np.hypot((lat - PLANT[0]) * 111.0, (lon - PLANT[1]) * 111.0 * np.cos(np.deg2rad(PLANT[0])))


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


# --------------------------------------------------------------------------- per-sounding work


def solar_tables():
    """Per band: solar wavenumber grid, absorption × continuum (production: continuum interpolated)."""
    sol = []
    with h5py.File(FILES["solar"], "r") as f:
        for ib in range(3):
            g = f[f"Solar/Absorption/Absorption_{ib + 1}"]
            nu, val = g["wavenumber"][...], g["spectrum"][...]
            c = f[f"Solar/Continuum/Continuum_{ib + 1}"]
            cont = np.interp(nu, c["wavenumber"][...], c["spectrum"][...])
            sol.append(dict(wvl=1e4 / nu, fsol=val * cont))
    return sol


def toa_solar(i: int) -> np.ndarray:
    """[3, 1016] TOA solar irradiance for sounding position i (fp_abs_coeff formula)."""
    g = ra._G
    fp = int(g["fp"][i])
    beta = g["v_sol"][i] / C_LIGHT
    mu = 1.0 / np.cos(np.deg2rad(g["sza"][i]))
    out = np.empty((3, 1016))
    for ib, (b, s) in enumerate(zip(g["bands"], g["sol"])):
        wv = s["wvl"] * (1.0 + beta)
        fs = s["fsol"] * (1.0 / g["dist_au"][i]) ** 2 / (1.0 + beta)
        wl, xx, yy = b["wl"][fp], b["xx"][fp], b["yy"][fp]
        start, end = ra.conv_indices(wl, xx, wv)
        xw = xx + wl[:, None]
        for c in range(len(wl)):
            w = np.interp(wv[start[c]:end[c]], xw[c], yy[c])
            out[ib, c] = np.dot(fs[start[c]:end[c]], w) / w.sum() * mu
    return out


def _work(idx):
    return [(i, split_one(i), toa_solar(i)) for i in idx]


# --------------------------------------------------------------------------- driver


def select_soundings():
    with h5py.File(FILES["l1b"], "r") as f:
        g = f["SoundingGeometry"]
        sid = g["sounding_id"][...].astype(np.int64)                      # [T, 8]
        lat, lon = g["sounding_latitude"][...], g["sounding_longitude"][...]
        sza, vza = g["sounding_solar_zenith"][...], g["sounding_zenith"][...]
        land = g["sounding_land_fraction"][...]
        v_sol = g["sounding_solar_relative_velocity"][...]
        dist = g["sounding_solar_distance"][...] / AU_M
        tsec = f["FrameHeader/frame_time_tai93"][...] if "FrameHeader/frame_time_tai93" in f else None
    d = km_from_plant(lat, lon)
    ok = np.isfinite(lat) & np.isfinite(lon) & (sid > 0) & (d < RADIUS_KM)
    tr, fp = np.nonzero(ok)
    frame_lat, frame_lon = np.nanmean(lat, axis=1), np.nanmean(lon, axis=1)
    i0 = np.nanargmin(km_from_plant(frame_lat, frame_lon))
    along = np.hypot((frame_lat - frame_lat[i0]) * 111.0,
                     (frame_lon - frame_lon[i0]) * 111.0 * np.cos(np.deg2rad(PLANT[0])))
    along = np.where(np.arange(len(along)) < i0, -along, along)
    return dict(sid=sid[tr, fp], track=tr, fp=fp, lat=lat[tr, fp], lon=lon[tr, fp], sza=sza[tr, fp],
                vza=vza[tr, fp], land=land[tr, fp] > 90, v_sol=v_sol[tr, fp], dist_au=dist[tr, fp],
                dist_km=d[tr, fp], along_km=along[tr], tsec=None if tsec is None else tsec[tr])


def compute_spectra(sel, workers):
    for k in ("l1b", "met", "cpr"):
        setattr(ra, k.upper(), FILES[k])
    n = len(sel["sid"])
    t0 = time.time()
    bands, h2o_meta = ra.load_tables()
    atm = ra.build_atmosphere(sel["sid"])
    ra._G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=sel["fp"], sza=sel["sza"], vza=sel["vza"],
                 sign=DOPPLER_SIGN, sol=solar_tables(), v_sol=sel["v_sol"], dist_au=sel["dist_au"])
    print(f"tables and atmosphere for {n} soundings in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    chunks = np.array_split(np.arange(n), max(1, n // 4))
    res = []
    with mp.get_context("fork").Pool(workers) as pool:
        for part in pool.imap(_work, chunks):
            res.extend(part)
    res.sort(key=lambda r: r[0])
    split = np.stack([r[1] for r in res]).astype(np.float32)              # [n, 3 bands, 3 comp, 1016]
    toa = np.stack([r[2] for r in res]).transpose(1, 0, 2).astype(np.float32) * _M1
    print(f"τ split and TOA solar in {time.time() - t0:.0f} s ({workers} workers)", flush=True)
    tau = split.sum(axis=2).transpose(1, 0, 2)                           # [3, n, 1016]
    tau_h2o = split[:, :, 1, :].transpose(1, 0, 2)

    rad = np.full((3, n, 1016), np.nan, np.float32)
    with h5py.File(FILES["l1b"], "r") as f:
        snr_coef = f["InstrumentHeader/snr_coef"][...]
        max_ms = f["InstrumentHeader/measureable_signal_max_observed"][...]
        for ib, b in enumerate(BANDS):
            full = f[f"SoundingMeasurements/{_L1B_RAD[b]}"]
            for i in range(n):
                rad[ib, i] = full[sel["track"][i], sel["fp"][i]]
    noise = np.stack([l1b_noise(rad[ib], snr_coef[ib][sel["fp"]], float(max_ms[ib])) for ib in range(3)])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rad / toa * np.pi
        t[(t <= 0) | (t > 1)] = np.nan
        ln_t = np.log(t).astype(np.float32)
        sig = (noise / rad).astype(np.float32)
    sig[~np.isfinite(ln_t)] = np.nan
    return tau, tau_h2o, ln_t, sig, atm


def plain_fit(tau, ln_t, sig):
    """Single-τ linear fit ln T = c − k1 τ on the same channels as the agreed model: the 'before' reference."""
    k1 = np.full(tau.shape[:2], np.nan)
    for ib in range(3):
        for i in range(tau.shape[1]):
            y, t, s = ln_t[ib, i].astype(float), tau[ib, i].astype(float), sig[ib, i].astype(float)
            ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0)
            if ok.sum() < 20:
                continue
            a = np.column_stack([np.ones(ok.sum()), -t[ok]])
            k1[ib, i] = np.linalg.lstsq(a, y[ok], rcond=None)[0][1]
    return k1


def lite_join(sid):
    with h5py.File(FILES["lite"], "r") as f:
        lsid = f["sounding_id"][...].astype(np.int64)
        pos = {s: i for i, s in enumerate(lsid)}
        idx = np.array([pos.get(s, -1) for s in sid])
        ok = idx >= 0

        def g(k):
            v = np.full(len(sid), np.nan)
            v[ok] = np.ma.filled(f[k][...][idx[ok]].astype(float), np.nan)
            return v
        out = dict(xco2_bc=g("xco2"), xco2_prior=g("xco2_apriori"), xco2_raw=g("Retrieval/xco2_raw"),
                   xco2_idp=g("Preprocessors/xco2_strong_idp"), qf=g("xco2_quality_flag"),
                   fs_rel=g("Retrieval/fs_rel"), lite_land=g("Sounding/land_fraction"))
    pq = pd.read_parquet(FILES["parquet"], columns=["fp_id", "cld_dist_km", "weighted_cloud_dist_km"])
    pq = pq.drop_duplicates("fp_id").set_index("fp_id")
    out["cld_dist_km"] = pq["cld_dist_km"].reindex(sid).values.astype(float)
    return out


def main():
    workers = int(os.environ.get("KOZ_WORKERS", "8"))
    sel = select_soundings()
    n = len(sel["sid"])
    print(f"{n} soundings within {RADIUS_KM:.0f} km of Kozienice; land {int(sel['land'].sum())}", flush=True)
    cache = OUT / f"{TAG}_spectra.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tau, tau_h2o, ln_t, sig, atm = z["tau"], z["tau_h2o"], z["ln_t"], z["sig"], z["atm"].item()
        print(f"spectra loaded from {cache.name}", flush=True)
    else:
        tau, tau_h2o, ln_t, sig, atm = compute_spectra(sel, workers)
        np.savez_compressed(cache, tau=tau, tau_h2o=tau_h2o, ln_t=ln_t, sig=sig, atm=np.array(atm, dtype=object))

    fm._G.update(tau=tau, tau_h2o=tau_h2o, ln_t=ln_t, sig=sig, land=sel["land"])
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(fm._chunk, np.array_split(np.arange(n), workers * 4))
    long = pd.DataFrame([r for p in parts for r in p])
    print(f"agreed-model fit in {time.time() - t0:.1f} s", flush=True)
    k1_plain = plain_fit(tau, ln_t, sig)

    w = lambda col: long.pivot(index="j", columns="band", values=col).reindex(range(n))  # noqa: E731
    k1, k1w, a1, f_o2a, k2, rms = w("k1"), w("k1_w"), w("a1"), w("f"), w("k2"), w("rms")
    lt = lite_join(sel["sid"])
    df = pd.DataFrame(dict(sounding_id=sel["sid"], fp=sel["fp"], lat=sel["lat"], lon=sel["lon"],
                           dist_km=sel["dist_km"], along_km=sel["along_km"], land=sel["land"],
                           sza=sel["sza"], vza=sel["vza"], v_inst=atm["v_inst"],
                           h2o_col=(atm["h2o"] * atm["dz"]).sum(axis=1), **lt))
    for b in BANDS:
        df[f"k1_{b}"], df[f"a1_{b}"], df[f"rms_{b}"] = k1[b].values, a1[b].values, rms[b].values
        df[f"k1plain_{b}"] = k1_plain[BANDS.index(b)]
    df["k1w_wco2"], df["k1w_sco2"], df["f_o2a"], df["k2_sco2"] = k1w["wco2"].values, k1w["sco2"].values, f_o2a["o2a"].values, k2["sco2"].values
    # column-mean prior from the CPr profile over the same layers that built τ: defined for every sounding
    dry = atm["o2"] / ra.O2MIX
    df["xco2_prior_cpr"] = 1e6 * (atm["co2"] * atm["dz"]).sum(axis=1) / (dry * atm["dz"]).sum(axis=1)
    pri = df.xco2_prior_cpr.values
    df["E0"] = pri * df.k1_sco2
    df["E1"] = pri * df.k1_sco2 / df.k1_o2a
    df["E1w"] = pri * df.k1_wco2 / df.k1_o2a
    df["E1_plain"] = pri * df.k1plain_sco2 / df.k1plain_o2a
    df["E0_plain"] = pri * df.k1plain_sco2
    df.to_parquet(OUT / f"{TAG}_per_sounding.parquet", index=False)

    # ---------------- evaluation
    core = df.dist_km <= CORE_KM
    bg = (df.dist_km >= BG_KM) & (df.qf == 0)
    bgall = df.dist_km >= BG_KM
    quantities = [("xco2_bc", "L2 xco2 (bias-corrected)"), ("xco2_raw", "L2 xco2_raw"), ("xco2_idp", "L2 strong IDP"),
                  ("xco2_prior", "Lite prior"), ("xco2_prior_cpr", "CPr column prior (used)"), ("E0", "agreed E0 = prior·k1_S"), ("E1", "agreed E1 = prior·k1_S/k1_O2A"),
                  ("E1w", "agreed E1w = prior·k1_W/k1_O2A"), ("E0_plain", "plain fit E0"), ("E1_plain", "plain fit E1")]
    rows = []
    inc_bc, inc_raw = (df.xco2_bc - pri).values, (df.xco2_raw - pri).values
    for col, lab in quantities:
        v = df[col].values
        inc = v - pri
        m_bg, m_core = bg & np.isfinite(v), core & np.isfinite(v)
        near = (df.dist_km < BG_KM) & np.isfinite(v) & np.isfinite(inc_bc)
        rec = dict(quantity=lab, col=col, n_bg=int(m_bg.sum()), n_core=int(m_core.sum()),
                   bg_median=np.nanmedian(v[m_bg]), bg_bias_vs_bc=np.nanmedian((v - df.xco2_bc)[m_bg]),
                   bg_bias_vs_prior=np.nanmedian(inc[m_bg]), bg_sd=np.nanstd(v[m_bg]), bg_rsd=robust_sd(v[m_bg]),
                   bg_rsd_vs_bc=robust_sd((v - df.xco2_bc)[m_bg]),
                   core_minus_bg=np.nanmedian(v[m_core]) - np.nanmedian(v[m_bg]),
                   core_max_minus_bg=np.nanmax(v[m_core]) - np.nanmedian(v[m_bg]) if m_core.any() else np.nan,
                   corr_inc_bc_near=np.corrcoef(inc[near], inc_bc[near])[0, 1] if near.sum() > 10 else np.nan,
                   slope_inc_bc_near=np.polyfit(inc_bc[near], inc[near], 1)[0] if near.sum() > 10 else np.nan,
                   corr_inc_raw_near=np.corrcoef(inc[near & np.isfinite(inc_raw)], inc_raw[near & np.isfinite(inc_raw)])[0, 1] if near.sum() > 10 else np.nan)
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{TAG}.csv", index=False)

    fmt = lambda x: "—" if not np.isfinite(x) else f"{x:+.2f}"  # noqa: E731
    lines = [f"# Kozienice plume, orbit 38200a_ND (2021-09-06): pseudo-XCO2 from the agreed model", "",
             f"{n} soundings within {RADIUS_KM:.0f} km of the plant (land {int(df.land.sum())}, Lite matches "
             f"{int(np.isfinite(df.xco2_bc).sum())}, QF0 {int((df.qf == 0).sum())}); core = within {CORE_KM:.0f} km "
             f"(n = {int(core.sum())}), background = beyond {BG_KM:.0f} km and QF0 (n = {int(bg.sum())}). "
             f"Cloud distance from the feature parquet: median {np.nanmedian(df.cld_dist_km):.1f} km. "
             f"Lite fs_rel median {np.nanmedian(df.fs_rel) * 100:.2f} % (validation only).", "",
             "Nothing is scaled. Columns: background median; bias vs xco2_bc and vs prior (background medians); "
             "SD / robust SD in the background; robust SD of (q − xco2_bc) in the background; core − background "
             "(medians) and max-in-core − background; correlation and slope of (q − prior) on (xco2_bc − prior) "
             "and correlation on (xco2_raw − prior) for soundings within 40 km.", "",
             "| quantity | bg median | bias vs bc | bias vs prior | bg SD | bg rSD | rSD vs bc | core − bg | max − bg | corr(bc) | slope(bc) | corr(raw) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lines.append(f"| {r.quantity} | {r.bg_median:.2f} | {fmt(r.bg_bias_vs_bc)} | {fmt(r.bg_bias_vs_prior)} | "
                     f"{r.bg_sd:.2f} | {r.bg_rsd:.2f} | {r.bg_rsd_vs_bc:.2f} | {fmt(r.core_minus_bg)} | "
                     f"{fmt(r.core_max_minus_bg)} | {fmt(r.corr_inc_bc_near)} | {fmt(r.slope_inc_bc_near)} | {fmt(r.corr_inc_raw_near)} |")
    # fit-term summary
    lines += ["", "## Fit terms (background medians, robust SD)", "", "| band | k1 | k1 rSD % | k1_w | a1 | f | k2 | rms | plain k1 | plain rSD % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for b in BANDS:
        v, vp = df[f"k1_{b}"][bg], df[f"k1plain_{b}"][bg]
        kw = df["k1w_" + b][bg].median() if b != "o2a" else np.nan
        lines.append(f"| {b} | {v.median():.4f} | {100 * robust_sd(v) / v.median():.2f} | {kw:.3f} | "
                     f"{df[f'a1_{b}'][bg].median():+.4f} | {df.f_o2a[bg].median():.4f} | {df.k2_sco2[bg].median():+.4f} | "
                     f"{df[f'rms_{b}'][bg].median():.4f} | {vp.median():.4f} | {100 * robust_sd(vp) / vp.median():.2f} |")
    # along-track profile, 5-km bins
    bins = np.arange(-RADIUS_KM, RADIUS_KM + 5, 5.0)
    prof = df.groupby(pd.cut(df.along_km, bins), observed=True)[["xco2_bc", "xco2_raw", "xco2_prior_cpr", "xco2_idp", "E0", "E1", "E1_plain"]].median()
    prof["n"] = df.groupby(pd.cut(df.along_km, bins), observed=True).size()
    near_tab = prof[(prof.index.map(lambda iv: abs(iv.mid)) <= 30)]
    lines += ["", "## Along-track 5-km medians within 30 km of closest approach (ppm)", "",
              "| along km | n | xco2_bc | xco2_raw | prior | IDP | E0 | E1 | plain E1 |", "|---|---|---|---|---|---|---|---|---|"]
    for iv, r in near_tab.iterrows():
        lines.append(f"| {iv.mid:+.0f} | {int(r.n)} | {r.xco2_bc:.2f} | {r.xco2_raw:.2f} | {r.xco2_prior_cpr:.2f} | {r.xco2_idp:.2f} | "
                     f"{r.E0:.2f} | {r.E1:.2f} | {r.E1_plain:.2f} |")
    # per-band fit terms along the plume: 5-km bins within 15 km vs background medians
    cols = [f"k1_{b}" for b in BANDS] + ["k1w_wco2", "k1w_sco2", "a1_o2a", "a1_wco2", "a1_sco2", "f_o2a", "k2_sco2", "xco2_bc", "cld_dist_km"]
    bgm = df.loc[bg, cols].median()
    lines += ["", "## Plume signature per band: 5-km bin medians minus background medians", "",
              "k1 contrasts in % (a CO2 enhancement of Δ ppm should give SCO2 and WCO2 k1 up by Δ/prior with O2A flat; "
              f"background prior {np.nanmedian(pri[bg]):.1f} ppm, so 1 % = {np.nanmedian(pri[bg]) / 100:.1f} ppm); "
              "k1_w, a1, f, k2 absolute; Δbc in ppm.", "",
              "| along km | n (Lite) | Δk1 O2A % | Δk1 WCO2 % | Δk1 SCO2 % | Δk1_w W | Δk1_w S | Δa1 O2A | Δa1 W | Δa1 S | Δf | Δk2 S | Δbc ppm | cld km |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for iv, g in df.groupby(pd.cut(df.along_km, np.arange(-15, 20, 5.0)), observed=True):
        m = g[cols].median()
        d = m - bgm
        lines.append(f"| {iv.mid:+.0f} | {len(g)} ({int(np.isfinite(g.xco2_bc).sum())}) | "
                     + " | ".join(f"{100 * d[f'k1_{b}'] / bgm[f'k1_{b}']:+.2f}" for b in BANDS)
                     + f" | {d.k1w_wco2:+.3f} | {d.k1w_sco2:+.3f} | {d.a1_o2a:+.4f} | {d.a1_wco2:+.4f} | {d.a1_sco2:+.4f} | "
                     f"{d.f_o2a:+.4f} | {d.k2_sco2:+.4f} | {d.xco2_bc:+.2f} | {m.cld_dist_km:.1f} |")
    # the plume soundings themselves: Lite xco2_bc − background > 2 ppm
    pl = df[(df.xco2_bc - bgm.xco2_bc > 2) & (df.dist_km < 30)]
    if len(pl):
        d = pl[cols].median() - bgm
        lines += ["", f"Plume soundings (xco2_bc > background + 2 ppm, within 30 km; n = {len(pl)}): Δbc {d.xco2_bc:+.2f} ppm, "
                  + ", ".join(f"Δk1 {b} {100 * d[f'k1_{b}'] / bgm[f'k1_{b}']:+.2f} %" for b in BANDS)
                  + f", Δk1_w W {d.k1w_wco2:+.3f}, S {d.k1w_sco2:+.3f}, Δa1 S {d.a1_sco2:+.4f}, Δf {d.f_o2a:+.4f}; "
                  + f"E0 − bg {np.nanmedian(pl.E0) - np.nanmedian(df.E0[bg]):+.2f} ppm, E1 − bg {np.nanmedian(pl.E1) - np.nanmedian(df.E1[bg]):+.2f} ppm, "
                  + f"E1w − bg {np.nanmedian(pl.E1w) - np.nanmedian(df.E1w[bg]):+.2f} ppm, raw − bg {np.nanmedian(pl.xco2_raw) - np.nanmedian(df.xco2_raw[bg]):+.2f} ppm."]
    lines += ["", f"Lite prior vs CPr column prior: median difference {np.nanmedian(df.xco2_prior - df.xco2_prior_cpr):+.2f} ppm, "
              f"rSD {robust_sd(df.xco2_prior - df.xco2_prior_cpr):.2f} ppm."]
    (OUT / f"{TAG}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = prof.index.map(lambda iv: iv.mid).values.astype(float)
    for col, ls, lab in (("xco2_bc", "k.-", "L2 xco2 (bc)"), ("xco2_raw", "k.--", "L2 xco2_raw"), ("xco2_prior_cpr", "k:", "prior (CPr column)"),
                         ("xco2_idp", "C3.-", "L2 strong IDP"), ("E1", "C0.-", "agreed E1 = prior·k1_S/k1_O2A"), ("E0", "C2.-", "agreed E0 = prior·k1_S")):
        axes[0].plot(x, prof[col], ls, ms=3, lw=0.9, label=lab)
    axes[0].set_ylabel("XCO2 (ppm), 5-km medians"); axes[0].legend(fontsize=7, ncol=3); axes[0].axvline(0, color="0.6", lw=0.5)
    axes[1].plot(df.along_km, df.xco2_bc, "k.", ms=3, label="L2 xco2 (bc)")
    axes[1].plot(df.along_km, df.E1, "C0.", ms=3, alpha=0.6, label="agreed E1")
    axes[1].plot(df.along_km, df.xco2_prior_cpr, "k:", lw=0.8, label="prior")
    axes[1].set_ylabel("XCO2 (ppm), per sounding"); axes[1].legend(fontsize=7); axes[1].axvline(0, color="0.6", lw=0.5)
    lo, hi = np.nanpercentile(df.xco2_bc, [1, 99])
    axes[1].set_ylim(lo - 8, hi + 8)
    for b, c in zip(BANDS, ("C1", "C2", "C3")):
        axes[2].plot(df.along_km, df[f"k1_{b}"], ".", color=c, ms=3, alpha=0.6, label=f"k1 {b}")
    axes[2].set_ylabel("agreed-model k1"); axes[2].set_xlabel("along-track distance from closest approach (km)")
    axes[2].legend(fontsize=7); axes[2].axvline(0, color="0.6", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}.png", dpi=130)


if __name__ == "__main__":
    main()
