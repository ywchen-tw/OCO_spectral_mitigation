"""Round 3: physical PPDF retrieval (phys_ppdf.py) on slab-MC scenes and on one real date.

Stages
  slab     regenerate the round-2 slab-MC TEST scenes (slab_mc.py, seed 20261101, n_train 4000,
           n_test 2000; same RNG sequence, so identical scenes) keeping the spectra, fit them with
           the src _am functions, and run the physical retrieval (noise-free and noisy).  No training.
  real     one date (default 2018-07-01): far-cloud (> 20 km) QF0 soundings, up to --max-per-sfc
           per surface, matched to fp_tau_combined.h5 by sounding id; retrieval per sounding.
  summary  tables -> results_phys/summary_phys.md (+ csv)

    PYTHONPATH=src python idea_test/ppdf_direct_xco2/run_phys.py slab --workers 32
    PYTHONPATH=src python idea_test/ppdf_direct_xco2/run_phys.py real --date 2018-07-01 --workers 32
    PYTHONPATH=src python idea_test/ppdf_direct_xco2/run_phys.py summary --date 2018-07-01

xco2_bc is used only for scoring.  Lite aerosol fields are carried only as diagnostics.
No ref_/r15_/r05_ statistic is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phys_ppdf as pp  # noqa: E402
import slab_mc as sm  # noqa: E402

OUT = HERE / "results_phys"
RD = Path("/scratch/alpine/yuch8913/oco/spec_anal/results_doppler")
COMBINED = RD / "csv_collection" / "combined_2016_2020_dates.parquet"
COMPACT = Path("/projects/yuch8913/code-server/tmp/claude-777013/-projects-yuch8913-OCO-spectral-mitigation/"
               "dbbff218-f8c4-4618-9e1f-97576c0dfb85/scratchpad/ppdf_test/v2/compact")
FOLD_DIR = RD / "model_deep_ensemble" / "am"
SLAB_SEED, N_TRAIN, N_TEST = 20261101, 4000, 2000


def rsd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 20 else np.nan


def flat(res, prefix=""):
    """Retrieval dict -> flat row."""
    row = {prefix + k: v for k, v in res.items() if np.isscalar(v)}
    for key in ("pull", "resid", "F"):
        if key in res:
            for n, v in zip(pp.OBS_NAMES, res[key]):
                row[f"{prefix}{key}_{n}"] = float(v)
    if "A_row_g" in res:
        for s, v in zip(pp.STATE, res["A_row_g"]):
            row[f"{prefix}Ag_{s}"] = float(v)
    return row


def ix(name):
    return pp.OBS_NAMES.index(name)


def fit_all(sc, ys):
    return {b: pp.fit_band(sc, b, ys[b]) for b in pp.BANDS}


def obs_vec(fits):
    return np.array([fits[b].get(k, np.nan) for b, k in pp.OBS])


# ── stage: slab ─────────────────────────────────────────────────────────────────

def slab_scene(args):
    """Verbatim copy of slab_mc.scene's RNG sequence and radiative transfer, returning the
    spectra (clean and noisy) instead of only the fit features."""
    i, g_co2, s, seed = args
    rng = np.random.default_rng(seed)
    mu0 = np.cos(np.radians(s["sza"]))
    muv = np.cos(np.radians(s["vza"]))
    amf = 1 / mu0 + 1 / muv
    land = rng.random() < 0.6
    cloudy = rng.random() < 0.3
    tau_a = float(np.exp(rng.normal(np.log(0.08), 0.8)))
    ang = rng.uniform(0.2, 1.5)
    if rng.random() < 0.3:
        sb = rng.uniform(0.5, 0.85)
        st = sb - rng.uniform(0.1, 0.2)
    else:
        sb, st = 1.0, rng.uniform(0.75, 0.92)
    tau_c = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0)))) if cloudy else 0.0
    cb = rng.uniform(0.3, 0.9)
    ct = cb - rng.uniform(0.05, 0.15)
    if land:
        a_o = rng.uniform(0.05, 0.35)
        alb = {"o2a": a_o, "wco2": a_o * rng.uniform(0.8, 1.8)}
        alb["sco2"] = alb["wco2"] * rng.uniform(0.7, 1.2)
    else:
        a = rng.uniform(0.02, 0.06)
        alb = {"o2a": a, "wco2": a * rng.uniform(0.9, 1.1), "sco2": a * rng.uniform(0.9, 1.1)}
    omega_a = rng.uniform(0.88, 0.99)
    wl = {"o2a": 0.765, "wco2": 1.61, "sco2": 2.06}
    ps = s["ps"]
    truth = dict(i=i, g_co2=g_co2, land=int(land), cloudy=int(cloudy), tau_a=tau_a, ang=ang,
                 sig_top=st, sig_bot=sb, tau_c=tau_c, omega_a=omega_a, sza=s["sza"], vza=s["vza"],
                 **{f"alb_{b}": alb[b] for b in sm.BANDS})
    bands, y_clean, y_noisy = {}, {}, {}
    for b in sm.BANDS:
        ta = tau_a * (wl[b] / 0.765) ** (-ang)
        layers = [(st, sb, ta, 0.7, True)] + ([(ct, cb, tau_c, 0.85, False)] if cloudy else [])
        contrib = sm.run_mc(layers, mu0, muv, rng)
        tau = s[f"{b}_tau"].astype(float)
        fh = s[f"{b}_fh"].astype(float)
        tw = tau * fh
        td = tau - tw
        sr = np.clip(np.nan_to_num(s[f"{b}_pabs"] / ps, nan=0.5), 0.4, 0.93)
        m_ch = np.clip((2 * sr - 1) / (1 - sr), sm.M_GRID[0], sm.M_GRID[-1])
        gscale = g_co2 if b != "o2a" else 1.0
        tau_true_vert = (td * gscale + tw) / amf
        slope = rng.uniform(-0.05, 0.05)
        alb_ch = np.clip(alb[b] * (1 + slope * sm.X), 1e-3, 1)
        R = sm.reflectance(contrib, tau_true_vert, m_ch, alb_ch, omega_a)
        yc = np.log(np.maximum(R, 1e-30))
        noise = rng.normal(size=R.size) / (sm.SNR[b] * rng.uniform(0.6, 1.4)) \
            / np.sqrt(np.maximum(R / R.max(), 1e-4)) / np.sqrt(sm.CH_STEP)
        bands[b] = dict(tau=tau, fh=fh, fr=np.zeros_like(tau), m=m_ch, x=sm.X)
        y_clean[b], y_noisy[b] = yc, yc + noise
    return truth, bands, y_clean, y_noisy, mu0, muv, land


def slab_one(args):
    truth, bands, yc, yn, mu0, muv, land = slab_scene(args)
    # slab geometry of the first scattering: photon azimuth 0, sensor azimuth 0 (upward)
    cos_sca = np.sqrt(1 - mu0 ** 2) * np.sqrt(1 - muv ** 2) - mu0 * muv
    K = sm.N_PHOT / np.pi
    row = dict(truth)
    for tag, ys in (("", yc), ("n_", yn)):
        # observed fits (src _am functions) on the scene's channel set
        sc0 = pp.make_scene(mu0, muv, cos_sca, K, land, True,
                            {b: dict(bands[b], c_obs=0.0) for b in pp.BANDS})
        # make_scene sorts by tau: sort the spectra the same way
        ys_s = {b: ys[b][sc0["b"][b]["order"]] for b in pp.BANDS}
        fits = fit_all(sc0, ys_s)
        if tag == "n_":
            fits_noisy = fits
        else:
            fits_clean = fits
    for tag, fits in (("", fits_clean), ("n_", fits_noisy)):
        for b in pp.BANDS:
            for k in ("k1_am", "k1w_am", "k2diag_am", "f_am", "c_am", "a1_am", "rms_am"):
                row[f"{tag}{b}_{k}"] = fits[b].get(k, np.nan)
    rms = {b: fits_noisy[b]["rms_am"] for b in pp.BANDS}
    for tag, fits in (("", fits_clean), ("n_", fits_noisy)):
        sc = pp.make_scene(mu0, muv, cos_sca, K, land, True,
                           {b: dict(bands[b], c_obs=fits[b]["c_am"]) for b in pp.BANDS})
        try:
            S = pp.formal_cov(sc, fits_noisy, rms)
            Sy = pp.ALPHA ** 2 * S + np.diag(pp.FLOOR ** 2)
            y = obs_vec(fits)
            if not np.all(np.isfinite(y)):
                raise ValueError("observed fit failed")
            res = pp.retrieve(sc, y, Sy)
            row.update(flat(res, "r_" + tag))
            row[f"r_{tag}sig_formal_sco2_k1"] = float(np.sqrt(S[ix("sco2_k1_am"), ix("sco2_k1_am")]))
        except Exception as exc:  # noqa: BLE001
            row[f"r_{tag}error"] = repr(exc)
    return row


def slab_jobs(bank):
    rng = np.random.default_rng(SLAB_SEED)
    n = N_TRAIN + N_TEST
    g = np.where(np.arange(n) < N_TRAIN, 1.0, rng.uniform(0.99, 1.01, n))
    return [(i, g[i], bank[rng.integers(len(bank))], int(rng.integers(2 ** 31))) for i in range(n)]


def stage_slab(args):
    OUT.mkdir(parents=True, exist_ok=True)
    bank = sm.load_tau_bank()
    jobs = slab_jobs(bank)[N_TRAIN:]
    if args.n_scenes:
        jobs = jobs[:args.n_scenes]
    print(f"slab: {len(jobs)} test scenes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(slab_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    # reproduction check against round 2's slab_mc_scenes.csv
    ref = HERE / "results_2016_2020_v2" / "slab_mc_scenes.csv"
    if ref.exists():
        r = pd.read_csv(ref, usecols=["i", "g_co2", "o2a_k1_am", "sco2_k1_am", "n_sco2_k1_am"]).set_index("i")
        j = df.set_index("i").join(r, rsuffix="_r2", how="inner")
        print("reproduction max|diff|: g", np.nanmax(np.abs(j.g_co2 - j.g_co2_r2)),
              "o2a_k1", np.nanmax(np.abs(j.o2a_k1_am - j.o2a_k1_am_r2)),
              "sco2_k1", np.nanmax(np.abs(j.sco2_k1_am - j.sco2_k1_am_r2)),
              "n_sco2_k1", np.nanmax(np.abs(j.n_sco2_k1_am - j.n_sco2_k1_am_r2)), flush=True)
    df.to_parquet(OUT / (f"{args.out_name}.parquet" if args.out_name else f"slab_phys_scenes{args.tag}.parquet"), index=False)
    print(slab_table(df).to_string(index=False), flush=True)


SLAB_SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    "clear (no cloud, tau_a<0.1)": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1)).to_numpy(),
    "cloudy": lambda d: (d.cloudy == 1).to_numpy(),
    "land": lambda d: (d.land == 1).to_numpy(),
    "ocean": lambda d: (d.land == 0).to_numpy(),
    "clear land": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1) & (d.land == 1)).to_numpy(),
    "clear ocean": lambda d: ((d.cloudy == 0) & (d.tau_a < 0.1) & (d.land == 0)).to_numpy(),
}


def slab_table(df):
    rows = []
    for tag, noise in (("", "noise-free"), ("n_", "noisy")):
        gh = df.get(f"r_{tag}g")
        if gh is None:
            continue
        err = (gh - df.g_co2).to_numpy() * 400.0
        conv = df.get(f"r_{tag}converged", pd.Series(False, index=df.index)).fillna(False).astype(bool).to_numpy()
        for sub, fn in SLAB_SUBSETS.items():
            m = fn(df) & np.isfinite(err)
            e = err[m]
            if e.size < 10:
                continue
            rows.append(dict(noise=noise, subset=sub, n=int(m.sum()), conv=float(conv[m].mean()),
                             rsd_ppm=rsd(e), sd_ppm=float(np.std(e)), bias_ppm=float(np.mean(e)),
                             med_bias_ppm=float(np.median(e)),
                             corr_ghat_g=corr(gh[m], df.g_co2[m]),
                             med_dof_g=float(np.nanmedian(df[f"r_{tag}dof_g"][m])),
                             med_sig_g_ppm=float(np.nanmedian(df[f"r_{tag}sig_g"][m]) * 400),
                             med_chi2_obs=float(np.nanmedian(df[f"r_{tag}chi2_obs"][m])),
                             med_ms=float(np.nanmedian(df[f"r_{tag}ms"][m]))))
    return pd.DataFrame(rows).round(3)


# ── stage: real ─────────────────────────────────────────────────────────────────

LITE_DIAG = ["aod_total", "aod_dust", "aod_water", "aod_ice", "aod_sulfate", "aod_seasalt", "aod_oc",
             "aod_bc", "aod_strataer", "dust_height", "water_height", "ice_height", "h2o_scale",
             "psfc", "alb_o2a", "alb_wco2", "alb_sco2"]


def load_real(date, max_per_sfc, seed=0):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(COMBINED)
    i = f.schema_arrow.names.index("date")
    rg = {f.metadata.row_group(g).column(i).statistics.min.decode(): g for g in range(f.metadata.num_row_groups)}
    cols = ["date", "time", "orbit_id", "fp", "fp_id", "lat", "lon", "sza", "vza", "saa", "vaa",
            "psfc_prior", "sfc_type", "xco2_qf", "cld_dist_km", "xco2_apriori", "xco2_bc", "operation_mode"]
    for b in pp.BANDS:
        cols += [f"{b}_{k}" for k in ("k1_am", "k1w_am", "k2diag_am", "c_am", "a1_am", "rms_am", "ok_am",
                                      "nchan", "sig_k1_am", "sig_k2diag_am", "sig_k1w_am")]
    cols += ["o2a_f_am", "o2a_sig_f_am", "sco2_a2_am"] + LITE_DIAG
    d = f.read_row_group(rg[date], columns=cols).to_pandas()
    for c in ("date", "orbit_id"):
        d[c] = d[c].astype(str).str.replace("b'", "").str.replace("'", "")
    far = d[(d.cld_dist_km > 20) & (d.xco2_qf == 0)]
    far = far[(far[[f"{b}_ok_am" for b in pp.BANDS]] == 1).all(axis=1)]
    far = far[(far[[f"{b}_nchan" for b in pp.BANDS]] == 1014).all(axis=1)]
    parts = [g.sample(min(len(g), max_per_sfc), random_state=seed) for _, g in far.groupby("sfc_type")]
    return pd.concat(parts).sort_values(["orbit_id", "fp_id"]).reset_index(drop=True)


def real_tau(df, date):
    """{fp_id: dict(band arrays)} for channels 1..1014 read from each orbit's tau file."""
    import h5py
    out = {}
    for orb, g in df.groupby("orbit_id"):
        path = RD / date / orb / "fp_tau_combined.h5"
        if not path.exists():
            print(f"  missing tau file {path}", flush=True)
            continue
        with h5py.File(path, "r") as f:
            sid = f["sounding_id"][...].astype(np.int64)
            pos = pd.Series(np.arange(len(sid)), index=sid)
            want = g.fp_id.astype(np.int64)
            ok = want.isin(pos.index)
            rows = np.sort(pos[want[ok]].to_numpy())
            dat = {}
            for b in pp.BANDS:
                for k, name in (("tau", "tau_output"), ("fh", "frac_h2o"), ("fr", "frac_ray"), ("pabs", "p_abs_hpa")):
                    dat[(b, k)] = f[f"{b}_{name}"][rows, 1:-1]
            for r_i, r in enumerate(rows):
                out[int(sid[r])] = {key: v[r_i] for key, v in dat.items()}
        print(f"  {orb}: {ok.sum()}/{len(g)} matched", flush=True)
    return out


X_FULL = (np.arange(1, 1015) - 507.5) / 507.5


def real_one(args):
    row, tau = args
    out = {"fp_id": row["fp_id"]}
    try:
        mu0, muv = np.cos(np.radians(row["sza"])), np.cos(np.radians(row["vza"]))
        cos_sca = -mu0 * muv - np.sin(np.radians(row["sza"])) * np.sin(np.radians(row["vza"])) * \
            np.cos(np.radians(row["saa"] - row["vaa"]))
        land_fit = bool(np.isfinite(row["sco2_a2_am"]))
        ps = row["psfc_prior"]
        bands = {b: dict(tau=tau[(b, "tau")], fh=tau[(b, "fh")], fr=tau[(b, "fr")],
                         m=pp.m_profile(tau[(b, "pabs")], ps), x=X_FULL, c_obs=row[f"{b}_c_am"])
                 for b in pp.BANDS}
        sc = pp.make_scene(mu0, muv, cos_sca, mu0 ** 2, land_fit, False, bands)
        obsfit = {b: {k: row.get(f"{b}_{k}", np.nan) for k in ("c_am", "a1_am", "k1_am", "f_am", "k2diag_am")}
                  for b in pp.BANDS}
        rms = {b: row[f"{b}_rms_am"] for b in pp.BANDS}
        S = pp.formal_cov(sc, obsfit, rms)
        y = np.array([row[f"{b}_{k}"] for b, k in pp.OBS], float)
        Sy = pp.ALPHA ** 2 * S + np.diag(pp.FLOOR ** 2)
        res = pp.retrieve(sc, y, Sy)
        out.update(flat(res, "r_"))
        # check of the reconstructed formal sigmas against the stored sig_*
        sd = np.sqrt(np.diag(S))
        out["sigchk_sco2_k1"] = float(sd[ix("sco2_k1_am")] / row["sco2_sig_k1_am"])
        out["sigchk_o2a_k1"] = float(sd[ix("o2a_k1_am")] / row["o2a_sig_k1_am"])
        out["sigchk_wco2_k1"] = float(sd[ix("wco2_k1_am")] / row["wco2_sig_k1_am"])
        if "o2a_f_am" in pp.OBS_NAMES:
            out["sigchk_o2a_f"] = float(sd[ix("o2a_f_am")] / row["o2a_sig_f_am"])
            out["corr_formal_o2a_k1_f"] = float(S[ix("o2a_k1_am"), ix("o2a_f_am")]
                                                / (sd[ix("o2a_k1_am")] * sd[ix("o2a_f_am")]))
        # forward model at the prior state (diagnostic)
        F0 = pp.forward(sc, pp.XA.copy())
        for n, v in zip(pp.OBS_NAMES, F0):
            out[f"Fprior_{n}"] = float(v)
    except Exception as exc:  # noqa: BLE001
        out["r_error"] = repr(exc)
    return out


def stage_real(args):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_real(args.date, args.max_per_sfc)
    print(f"{args.date}: {len(df)} far-cloud QF0 soundings sampled "
          f"({df.groupby('sfc_type').size().to_dict()})", flush=True)
    tau = real_tau(df, args.date)
    df = df[df.fp_id.astype(np.int64).isin(list(tau))].reset_index(drop=True)
    jobs = [(r._asdict() if hasattr(r, "_asdict") else r, tau[int(r["fp_id"])])
            for r in df.to_dict("records")]
    print(f"retrieving {len(jobs)} soundings", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(real_one, jobs, chunksize=4))
    r = pd.DataFrame(res)
    out = df.merge(r, on="fp_id", how="left")
    out.to_parquet(OUT / (f"{args.out_name}.parquet" if args.out_name else f"real_{args.date}{args.tag}.parquet"), index=False)
    print("done", flush=True)


# ── stage: summary ──────────────────────────────────────────────────────────────

def round2_estimates(df, date):
    """E0_am, E2cP_am_hgb (round-2 compact frame, held-out date block) and OB_E2cP_am_hgb
    (training-date blend coefficients of the fold that holds this date) on the same soundings."""
    p = COMPACT / f"{date}.parquet"
    if not p.exists():
        return df
    c = pd.read_parquet(p, columns=["time", "fp", "est_E0_am", "est_E2cP_am_hgb"])
    c = c.drop_duplicates(["time", "fp"])
    out = df.merge(c, on=["time", "fp"], how="left")
    fold = None
    for k in range(5):
        d = json.loads((FOLD_DIR / f"fm1620k5_ocean_full_sig_f{k}" / "training_dates.json").read_text())
        if date in d["held_dates"]:
            fold = k
    co = pd.read_csv(HERE / "results_2016_2020_v2" / "opblend_coefs.csv")
    co = co[(co.est == "OB_E2cP_am_hgb") & (co.fold == fold)].set_index("sfc_type")
    pri = out.xco2_apriori.to_numpy(float)
    a = out.sfc_type.map(co["a"]).to_numpy(float)
    w = out.sfc_type.map(co["w_E2cP_am_hgb"]).to_numpy(float)
    out["est_OB_E2cP_am_hgb"] = pri + a + w * (out.est_E2cP_am_hgb.to_numpy(float) - pri)
    out.attrs["fold"] = fold
    return out


def score(est, bc, pri):
    ok = np.isfinite(est) & np.isfinite(bc)
    e, b, p = est[ok], bc[ok], pri[ok]
    r_bc = rsd(e - b)
    r_same = rsd(b - p)
    bias = float(np.mean(e - b))
    return dict(n=int(ok.sum()), rsd_vs_bc=r_bc, bias_vs_bc=bias, total=float(np.hypot(r_bc, bias)),
                rsd_vs_prior=rsd(e - p), rsd_bc_vs_prior=r_same,
                R2=1 - (r_bc / r_same) ** 2 if r_same > 0 else np.nan,
                corr_bcdev=corr(e - p, b - p) if rsd(e - p) > 0 else np.nan)


def stage_summary(args):
    L = [f"# Physical PPDF retrieval, round 3 ({args.date} + slab MC)", ""]
    for slab_p in sorted(OUT.glob("slab_phys_scenes*.parquet")):
        stag = slab_p.stem.replace("slab_phys_scenes", "")
        sl = pd.read_parquet(slab_p)
        t = slab_table(sl)
        t.to_csv(OUT / f"slab_phys_floor{stag}.csv", index=False)
        L += [f"## Slab MC{stag} (error of g-hat, ppm; no training; matched: {sorted(c[7:] for c in sl.columns if c.startswith('r_pull_'))})", "", t.to_markdown(index=False), ""]
        for tag in ("", "n_"):
            pc = [c for c in sl.columns if c.startswith(f"r_{tag}pull_")]
            if pc:
                L += [f"pull rsd / median ({'noisy' if tag else 'noise-free'}):", "",
                      pd.DataFrame({"rsd": sl[pc].apply(rsd), "median": sl[pc].median()}).round(3).to_markdown(), ""]
            sc = [f"r_{tag}{s}" for s in pp.STATE]
            tr = {"r_%sln_tauL" % tag: np.log(sl.tau_a)}
            L += [f"corr(retrieved ln tau_L, true ln tau_a) [{tag or 'clean'}]: "
                  f"{corr(sl[f'r_{tag}ln_tauL'], np.log(sl.tau_a)):.2f}; "
                  f"corr(sigma_H, true layer mid sigma): {corr(sl[f'r_{tag}sigma_H'], 0.5 * (sl.sig_top + sl.sig_bot)):.2f}", ""]
    for rp in sorted(OUT.glob(f"real_{args.date}*.parquet")):
        rtag = rp.stem.replace(f"real_{args.date}", "")
        d = pd.read_parquet(rp)
        d = round2_estimates(d, args.date)
        d["est_phys"] = d.xco2_apriori * d.r_g
        conv = d.r_converged.fillna(False).astype(bool)
        rows, diag = [], []
        for s, g in d.groupby("sfc_type"):
            name = {0.0: "ocean", 1.0: "land"}[s]
            gc = g[g.r_converged.fillna(False).astype(bool)]
            pri, bc = gc.xco2_apriori.to_numpy(float), gc.xco2_bc.to_numpy(float)
            off = float(np.nanmean(bc - gc.est_phys))     # same-date diagnostic offset
            series = {"prior only": pri,
                      "PHYS (raw, uncalibrated)": gc.est_phys.to_numpy(float),
                      "PHYS + same-date offset [diagnostic only]": gc.est_phys.to_numpy(float) + off}
            # analytic re-prior: sigma_a(g) = 0.25 % (effective measurement from the weak-prior posterior)
            sp2 = gc.r_sig_g.to_numpy(float) ** 2
            sa2 = pp.SA_SD[0] ** 2
            sm2 = 1 / np.clip(1 / sp2 - 1 / sa2, 1e-12, None)
            gm = 1 + (gc.r_g.to_numpy(float) - 1) * (sm2 + sa2) / sa2    # unshrunk measurement of g
            st2 = 0.0025 ** 2
            gt = 1 + (gm - 1) * st2 / (st2 + sm2)
            series["PHYS, sigma_a(g)=0.25 % (analytic re-prior)"] = pri * gt
            for c, lab in (("est_E0_am", "E0_am (round 1/2)"), ("est_E2cP_am_hgb", "E2cP_am_hgb (round 2, ML)"),
                           ("est_OB_E2cP_am_hgb", "OB_E2cP_am_hgb (round 2, ML + blend)")):
                if c in gc:
                    series[lab] = gc[c].to_numpy(float)
            for lab, v in series.items():
                rows.append(dict(surface=name, estimator=lab, **score(v, bc, pri)))
            ms = g.r_ms.to_numpy(float)
            diag.append(dict(surface=name, n=len(g), conv_rate=float(conv[g.index].mean()),
                             med_iter=float(gc.r_n_iter.median()), med_nfwd=float(gc.r_n_fwd.median()),
                             chi2_med=float(gc.r_chi2_obs.median()), chi2_q90=float(gc.r_chi2_obs.quantile(0.9)),
                             chi2_per_obs_med=float(gc.r_chi2_obs.median() / sum(c.startswith("r_pull_") for c in gc.columns)),
                             dof_g_med=float(gc.r_dof_g.median()), dof_g_q10=float(gc.r_dof_g.quantile(0.1)),
                             dof_g_q90=float(gc.r_dof_g.quantile(0.9)), dof_total_med=float(gc.r_dof_total.median()),
                             sig_g_ppm_med=float(gc.r_sig_g.median() * 400), same_date_offset_ppm=off,
                             ms_med=float(np.nanmedian(ms)), ms_mean=float(np.nanmean(ms)),
                             tauL_med=float(np.exp(gc.r_ln_tauL).median()), sigmaH_med=float(gc.r_sigma_H.median()),
                             alpha_med=float(gc.r_alpha.median()), h_med=float(gc.r_h.median()),
                             s_p_med=float(gc.r_s_p.median()), g_med=float(gc.r_g.median()),
                             frac_tauL_at_bound=float((gc.r_ln_tauL <= pp.LB[3] + 1e-6).mean() + (gc.r_ln_tauL >= pp.UB[3] - 1e-6).mean()),
                             corr_tauL_aod_total=corr(gc.r_ln_tauL, np.log(gc.aod_total.clip(1e-4))),
                             spearman_tauL_aod=float(pd.Series(gc.r_ln_tauL.to_numpy()).corr(pd.Series(gc.aod_total.to_numpy()), method="spearman")),
                             spearman_sigmaH_dust_height=float(pd.Series(gc.r_sigma_H.to_numpy()).corr(pd.Series(gc.dust_height.to_numpy()), method="spearman")),
                             spearman_h_h2o_scale=float(pd.Series(gc.r_h.to_numpy()).corr(pd.Series(gc.h2o_scale.to_numpy()), method="spearman")),
                             corr_sp_psfc_ratio=corr(gc.r_s_p, gc.psfc / gc.psfc_prior),
                             corr_g_minus1_vs_bc_minus_prior=corr(gc.r_g - 1, bc / pri - 1)))
        tab = pd.DataFrame(rows).round(3)
        dg = pd.DataFrame(diag).round(3)
        tab.to_csv(OUT / f"real_{args.date}{rtag}_table.csv", index=False)
        dg.to_csv(OUT / f"real_{args.date}{rtag}_diag.csv", index=False)
        L += [f"## Real far-cloud QF0 soundings, {args.date}{rtag} (matched: {[c[7:] for c in d.columns if c.startswith('r_pull_')]}; round-2 fold {d.attrs.get('fold')})", "",
              tab.to_markdown(index=False), "", "### Diagnostics", "", dg.T.to_markdown(), ""]
        pc = [c for c in d.columns if c.startswith("r_pull_")]
        rc = [c for c in d.columns if c.startswith("r_resid_")]
        fp = [f"Fprior_{c[7:]}" for c in pc]
        for s, g in d[conv].groupby("sfc_type"):
            names = [c[7:] for c in pc]
            obs = g[names].median().to_numpy()
            L += [f"### Observables, {'ocean' if s == 0 else 'land'} (medians)", "",
                  pd.DataFrame({"obs": obs, "F(prior state)": g[fp].median().to_numpy(),
                                "resid (obs-F) med": g[rc].median().to_numpy(),
                                "resid rsd": g[rc].apply(rsd).to_numpy(),
                                "pull rsd": g[pc].apply(rsd).to_numpy(),
                                "obs rsd": g[names].apply(rsd).to_numpy()},
                               index=names).round(4).to_markdown(), ""]
        sk = [c for c in d.columns if c.startswith("sigchk_") or c == "corr_formal_o2a_k1_f"]
        L += ["Reconstructed formal sigma / stored sig_* (median):", "", d[sk].median().round(3).to_markdown(), ""]
        if "r_error" in d:
            L += [f"errors: {d.r_error.notna().sum()}", ""]
    (OUT / "summary_phys.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["slab", "real", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-scenes", type=int, default=0)
    ap.add_argument("--date", default="2018-07-01")
    ap.add_argument("--max-per-sfc", type=int, default=1000)
    ap.add_argument("--exclude-obs", nargs="*", default=[],
                    help="observable names to drop from the match, e.g. o2a_f_am")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--sp-sd", type=float, default=None,
                    help="prior sigma of s_p (default 0.002); bounds widen to 0.85-1.15 when set")
    ap.add_argument("--out-name", default="", help="output parquet stem (overrides the default name)")
    args = ap.parse_args()
    pp.set_obs(args.exclude_obs)
    if args.sp_sd is not None:
        pp.SA_SD[1] = args.sp_sd
        pp.LB[1], pp.UB[1] = 0.85, 1.15
        print(f"s_p prior sigma {pp.SA_SD[1]}, bounds {pp.LB[1]}-{pp.UB[1]}", flush=True)
    print("matched observables:", pp.OBS_NAMES, flush=True)
    {"slab": stage_slab, "real": stage_real, "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
