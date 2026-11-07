"""
Stage 4b: agreed-model (AM) fits of the slab condition sweep, per scene.

For every scene under <OUT_DIR>/scenes/<scene>/slab_rad.h5 (the 1e8-photon
screening sweep of 2026-11-06), every surface and solver, every column x:
fit ln T(lambda) with the production AM estimator of
src/spectral/cumulant_fit (fitting.py schema doppler_am_t2_v2):

  O2A   : ln T = c + a1 x + ln[(1 - f) exp(-k1 tau) + f]        (k2 fixed 0;
          k2diag = free k2 of the refit with 1/2 k2 tau^2 in the exponent)
  WCO2  : ln T = c + a1 x - k1 tau                              (k2diag: + 1/2 k2 tau^2)
  SCO2  : ln T = c + a1 x [+ a2 x^2 land] - k1 tau + 1/2 k2 tau^2

Differences from the OCO-2 fit: (i) the slab OD file holds the TOTAL
(O2 or CO2 + H2O) layer optical depth, so the one-tau design is used (no
k1w term); (ii) the spectral coordinate x = (lambda - lambda_mid)/half-width
of the band window (BANDS[band]["wvl_um"]) replaces the channel index;
(iii) "land" for the SCO2 x^2 term = every surface except ocean/dark.
The unconstrained order-2 cumulant fit (k1_o2, k2_o2) and the production
constrained cumulant fit of fit_and_plot.py (k1_c, k2_c, order FIT_ORDER)
are stored alongside.  T = pi I / (mu0 F0); per-run fits give the
Monte-Carlo spread (std over NRUN runs, ddof=1).

Zones (per scene, per column): far field x < 5 km; cloud = nominal
CLOUD_X_KM columns; shadow = columns outside the cloud whose darkest-surface
3-D/IPA continuum ratio is < SHADOW_RATIO (IPA has no shadow, so the
darkening is the shadow by definition).  Slant-view scenes: the 3-D cloud is
displaced by the parallax h*tan(VZA) (about +4 km for sensor azimuth 270,
-4 km for 90) while the MCARaTS IPA cloud stays in its nadir column, so the
cloud mask is located from the darkest-surface 3-D continuum radiance and
ALL IPA profiles are rolled by that offset (cloud_offset_col attr) before
differencing -- the comparison is then 3-D vs parallax-aligned IPA.

Outputs:
  <OUT_DIR>/scenes/<scene>/slab_fit_am.h5
  <OUT_DIR>/scenes/_summary/scene_<scene>_am.png       per-scene quick-look
  <OUT_DIR>/scenes/_summary/summary_delta_<field>.png  3-D minus IPA, all scenes
  <OUT_DIR>/scenes/_summary/summary_am.csv / .md       zone statistics

--boxcar-km W averages the radiances (and, radiance-weighted, the path
histograms) over a W-km top-hat along x before fitting -- an OCO-2-footprint
view of the same scenes (2.25 km along-track); outputs get the suffix _fpW.

Run:  python workspace/rt_slab_sim/fit_scenes_am.py [--scenes a b ...] [--no-fig]
                                                    [--boxcar-km 2.25]
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from spectral.cumulant_fit import (_fit_am_o2a, _lin_se, _unconstrained_fit_stats,
                                   fit_spectral_model, get_design_matrix)

SCENES_DIR = cfg.SCENES_DIR          # results/rt_slab_sim/$SLAB_SCENES_SUBDIR (default scenes)
SUMMARY_DIR = os.path.join(SCENES_DIR, "_summary")
SHADOW_RATIO = 0.7
FAR_KM = 5.0
WATER_SURFACES = ("ocean", "dark")
AM_FIELDS = ("k1_am", "k2diag_am", "a1_am", "a2_am", "f_am", "c_am", "exp_int_am",
             "rms_am", "k1_o2", "k2_o2", "k1_c", "k2_c")
# fixed identity colours (never cycled): surface types across all scenes
SURFACE_COLORS = {"dark": "#1b6ca8", "bright": "#d1495b", "ocean": "#1b6ca8",
                  "veg": "#2a9d8f", "soil": "#e9a23b", "snow": "#7b5ea7",
                  "u3": "#1b6ca8", "u7": "#2a9d8f", "u12": "#e9a23b"}
SOLVER_COLORS = {"3d": "#d1495b", "ipa": "#1b6ca8"}
SCENE_ORDER = ("ref", "sza25", "sza70", "vza43_fwd", "vza43_back", "cloud_low",
               "cod2", "cod30", "rayleigh_off", "width1", "width15",
               "mie", "cer5", "cer20", "high_thin", "high_cod10", "aerosol", "coast",
               "o2a_sfc", "wco2", "sco2",
               "glint_cod0", "glint", "glint_sza15", "glint_sza45", "glint_cod2",
               "glint_cloud_low", "wco2_glint", "sco2_glint")


TAG = ""          # output-name suffix (set from --boxcar-km)


def boxcar(arr, width_cols, axis):
    """Circular top-hat running mean of fractional width (in columns)."""
    if width_cols <= 1.0:
        return arr
    h = 0.5 * width_cols
    kmax = int(np.ceil(h + 0.5))
    out = np.zeros_like(arr, dtype=np.float64)
    wsum = 0.0
    for k in range(-kmax, kmax + 1):
        w = np.clip(min(k + 0.5, h) - max(k - 0.5, -h), 0.0, 1.0)
        if w > 0:
            out += w * np.roll(arr, -k, axis=axis)
            wsum += w
    return out / wsum


# ─── fits ───────────────────────────────────────────────────────────────────
def fit_am(band, tau, x, y, land):
    """One-tau AM fit of one column; returns dict of AM_FIELDS (NaN on failure)."""
    out = {k: np.nan for k in AM_FIELDS}
    one = np.ones_like(tau)
    try:
        if band == "o2a":
            r = _fit_am_o2a(tau, x, y)
            if r["ok_am"] != 1.0:
                return out
            out.update({k: r[k] for k in ("k1_am", "k2diag_am", "a1_am", "f_am",
                                          "c_am", "rms_am")})
        elif band == "wco2":
            c, rms, _ = _lin_se([one, x, -tau], y)
            cd, _, _ = _lin_se([one, x, -tau, 0.5 * tau**2], y)
            out.update({"k1_am": c[2], "a1_am": c[1], "c_am": c[0], "rms_am": rms,
                        "k2diag_am": cd[3]})
        else:
            cols = [one, x] + ([x**2] if land else []) + [-tau, 0.5 * tau**2]
            c, rms, _ = _lin_se(cols, y)
            k = 3 if land else 2
            out.update({"k1_am": c[k], "k2diag_am": c[k + 1], "a1_am": c[1],
                        "a2_am": c[2] if land else np.nan, "c_am": c[0], "rms_am": rms})
        out["exp_int_am"] = np.exp(out["c_am"])
        coef, _, _ = _unconstrained_fit_stats(get_design_matrix(tau, 2), y)
        out["k1_o2"], out["k2_o2"] = coef[1], coef[2]
        p = fit_spectral_model(tau, y, cfg.BANDS[band]["fit_order"], smooth=False)
        out["k1_c"], out["k2_c"] = p[0], p[1]
    except (np.linalg.LinAlgError, ValueError):
        pass
    return out


def fit_columns(band, rad, toa, tau, x, mu0, land):
    """rad (nw, nx) -> dict field -> (nx,)."""
    nx = rad.shape[1]
    T = np.pi * rad / (mu0 * toa[:, None])
    res = {k: np.full(nx, np.nan) for k in AM_FIELDS}
    order = np.argsort(tau)
    for ix in range(nx):
        y = T[:, ix]
        if not np.all(y > 0):
            continue
        r = fit_am(band, tau[order], x[order], np.log(y[order]), land)
        for k in AM_FIELDS:
            res[k][ix] = r[k]
    return res


def plen_moments(h, mid):
    """(nw, nrun, nx, nbin) -> (nw, nx) mean and std of path [m], runs pooled."""
    h = h.mean(axis=1, dtype=np.float64)
    w = h.sum(axis=-1)
    w = np.where(w > 0, w, np.nan)
    m1 = (h * mid).sum(axis=-1) / w
    m2 = (h * mid**2).sum(axis=-1) / w
    return m1, np.sqrt(np.maximum(m2 - m1**2, 0.0))


# ─── zones ──────────────────────────────────────────────────────────────────
def cloud_offset(cont_ipa, nominal):
    """Columns by which the IPA continuum brightness pattern is shifted from
    the nominal cloud mask (circular cross-correlation)."""
    a = cont_ipa - cont_ipa.mean()
    b = nominal.astype(float) - nominal.mean()
    corr = [np.dot(a, np.roll(b, s)) for s in range(len(a))]
    return int(np.argmax(corr))


def zones(x_km, att, cont_3d, cont_ipa):
    """Return dict of boolean column masks and the detected offset."""
    nx = x_km.size
    x1, x2 = att["cloud_x_km"]
    nominal = (x_km >= x1) & (x_km < x2)
    if float(att.get("cloud_cod", 1.0)) <= 0:
        nominal[:] = False                      # cloud-free scene (coast)
    off = 0
    if att["sensor_zenith"] > 0 and nominal.any():
        # slant view: the 3-D cloud is displaced by the parallax h*tan(VZA)
        # while the IPA cloud stays in its nadir column -> locate the 3-D
        # cloud and shift the IPA profiles onto it (see roll_ipa)
        off = cloud_offset(cont_3d, nominal)
        if off > nx // 2:
            off -= nx
    cloud = np.roll(nominal, off)
    ratio = cont_3d / np.roll(cont_ipa, off)
    shadow = (~cloud) & (ratio < SHADOW_RATIO)
    far = x_km < FAR_KM
    return {"far": far, "cloud": cloud, "shadow": shadow,
            "near": ~(far | cloud | shadow)}, off, ratio


# ─── per scene ──────────────────────────────────────────────────────────────
def analyse_scene(scene, make_fig=True, boxcar_km=0.0):
    sdir = os.path.join(SCENES_DIR, scene)
    fn = os.path.join(sdir, "slab_rad.h5")
    if not os.path.exists(fn):
        print(f"[{scene}] no slab_rad.h5 -- skipped")
        return None
    res, att = {}, {}
    with h5py.File(fn, "r") as f:
        att = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else
                   (v if isinstance(v, str) else np.asarray(v))) for k, v in f.attrs.items()}
        band = att.get("band", "o2a")
        if isinstance(band, bytes):
            band = band.decode()
        att["band"] = band
        nx = int(att["nx"])
        x_km = (np.arange(nx) + 0.5) * float(att["dx_km"])
        mu0 = np.cos(np.deg2rad(float(att["sza"])))
        slant_all = f["slant_tau"][...]
        wvl_all = f["wvl_nm"][...]
        lo, hi = cfg.BANDS[band]["wvl_um"]
        x_all = (wvl_all * 1e-3 - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
        edges = np.linspace(att["plen_min_m"], att["plen_max_m"], int(att["plen_nbin"]) + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        i_cont = np.argsort(slant_all)[:cfg.N_CONTINUUM]
        surfaces = [s for s in f if isinstance(f[s], h5py.Group)]
        for surface in surfaces:
            land = surface not in WATER_SURFACES
            for solver in cfg.SOLVERS:
                key = f"{surface}/{solver}"
                if key not in f:
                    continue
                g = f[key]
                iw = g["iw"][...]
                tau, x = slant_all[iw], x_all[iw]
                toa = g["toa"][...]
                rad = np.asarray(g["rad"][...], dtype=np.float64).reshape(iw.size, nx)
                runs = np.asarray(g["rad_runs"][...], dtype=np.float64).reshape(iw.size, nx, -1)
                hist = g["plen_hist"][...]                      # (nw, nrun, nx, nbin)
                wcols = boxcar_km / float(att["dx_km"])
                if wcols > 1.0:
                    # footprint view: radiance-weighted top-hat along x
                    hist = (boxcar(hist * rad[:, None, :, None], wcols, axis=2)
                            / boxcar(rad, wcols, axis=1)[:, None, :, None])
                    rad = boxcar(rad, wcols, axis=1)
                    runs = boxcar(runs, wcols, axis=1)
                r = fit_columns(band, rad, toa, tau, x, mu0, land)
                r["rad_cont"] = rad[np.isin(iw, i_cont)].mean(axis=0)
                per_run = [fit_columns(band, runs[..., ir], toa, tau, x, mu0, land)
                           for ir in range(runs.shape[-1])]
                for k in AM_FIELDS:
                    r[f"{k}_std"] = np.std([p[k] for p in per_run], axis=0, ddof=1)
                m1, s1 = plen_moments(hist, mid)
                del hist
                r["plen_mean"], r["plen_std"] = m1, s1
                res[key] = r
                print(f"[{scene}] {key}: k1_am far-field mean "
                      f"{np.nanmean(r['k1_am'][x_km < FAR_KM]):.4f}", flush=True)

    s0 = min(surfaces, key=lambda s: np.nanmean(res[f"{s}/ipa"]["rad_cont"]))
    zn, off, ratio = zones(x_km, att, res[f"{s0}/3d"]["rad_cont"], res[f"{s0}/ipa"]["rad_cont"])
    if off:
        for key, r in res.items():
            if key.endswith("/ipa"):
                for name, val in r.items():
                    r[name] = np.roll(val, off, axis=-1)
    iw_mid = int(np.argmin(np.abs(slant_all[iw] - 1.0)))
    out = {"scene": scene, "att": att, "x_km": x_km, "tau": slant_all[iw], "wvl_nm": wvl_all[iw],
           "res": res, "zones": zn, "offset": off, "shadow_ratio": ratio,
           "surfaces": surfaces, "s0": s0, "iw_mid": iw_mid}
    zdesc = {k: (f"{x_km[m].min():.1f}-{x_km[m].max():.1f} km ({m.sum()} col)" if m.any()
                 else "none") for k, m in zn.items()}
    print(f"[{scene}] band {band} sza {att['sza']:.0f} vza {att['sensor_zenith']:.0f} "
          f"offset {off} col; zones {zdesc}")

    with h5py.File(os.path.join(sdir, f"slab_fit_am{TAG}.h5"), "w") as f:
        f["x_km"] = x_km
        f["tau"] = out["tau"]
        f["wvl_nm"] = out["wvl_nm"]
        f.attrs.update({k: v for k, v in att.items()})
        f.attrs["cloud_offset_col"] = off      # IPA groups are rolled by this (parallax alignment)
        f.attrs["shadow_ratio"] = SHADOW_RATIO
        f.attrs["darkest_surface"] = s0
        f.attrs["boxcar_km"] = boxcar_km
        for k, m in zn.items():
            f[f"zone_{k}"] = m
        for key, r in res.items():
            g = f.create_group(key)
            for name, val in r.items():
                g[name] = val
    if make_fig:
        plot_scene(out)
    return out


def _anom(v, far):
    return v - np.nanmedian(v[far])


def plot_scene(o):
    att, x_km, res, zn = o["att"], o["x_km"], o["res"], o["zones"]
    surfaces = o["surfaces"]
    band, scene = att["band"], o["scene"]
    rows = [("k1_am", r"$k_1^{\rm AM}$", False),
            ("k2diag_am", r"$k_2^{\rm diag}$", False),
            ("a1_am", r"$a_1$ (continuum slope)", False),
            ("exp_int_am", r"$e^{c}$ (eff. reflectance)", False),
            ("plen", "MC mean path anomaly @ τ≈1 (km)", True)]
    fig, axes = plt.subplots(len(rows), len(surfaces), figsize=(4.6 * len(surfaces), 2.4 * len(rows)),
                             sharex=True, squeeze=False)
    far = zn["far"]
    for jc, surface in enumerate(surfaces):
        # the scene's own surface table (cfg is imported for the base scene)
        scene_surfaces = cfg.SCENES.get(scene, {}).get("SURFACE_ALBEDOS", cfg.SURFACE_ALBEDOS)
        alb = scene_surfaces.get(surface, cfg.SURFACE_TYPES.get(surface, {}).get(band, np.nan))
        for solver in cfg.SOLVERS:
            r = res.get(f"{surface}/{solver}")
            if r is None:
                continue
            c = SOLVER_COLORS[solver]
            lbl = "3-D" if solver == "3d" else "IPA"
            for jr, (name, _, is_plen) in enumerate(rows):
                ax = axes[jr, jc]
                if is_plen:
                    pl = r["plen_mean"][o["iw_mid"]] / 1e3
                    ax.plot(x_km, _anom(pl, far), color=c, lw=1.6, label=lbl)
                else:
                    ax.plot(x_km, r[name], color=c, lw=1.6, label=lbl)
                    sd = r[f"{name}_std"]
                    ax.fill_between(x_km, r[name] - sd, r[name] + sd, color=c, alpha=0.2, lw=0)
        if isinstance(alb, dict):
            axes[0, jc].set_title(f"{surface} (glint, U {alb['wind_ms']:g} m/s)", fontsize=10)
        else:
            axes[0, jc].set_title(f"{surface} (albedo {alb:.2f})", fontsize=10)
        for jr, (name, lab, _) in enumerate(rows):
            ax = axes[jr, jc]
            for m, col in ((zn["cloud"], "0.80"), (zn["shadow"], "0.92")):
                for ix in np.flatnonzero(m):
                    ax.axvspan(x_km[ix] - 0.25 * att["dx_km"] * 2, x_km[ix] + 0.25 * att["dx_km"] * 2,
                               color=col, lw=0, zorder=0)
            if jc == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
        axes[0, jc].legend(fontsize=8, loc="upper right")
        axes[-1, jc].set_xlabel("x (km)")
    fig.suptitle(f"scene {o['scene']}  band {band}  SZA {att['sza']:.0f}°  VZA {att['sensor_zenith']:.0f}°  "
                 f"COD {att['cloud_cod']:.0f}  cloud {att['cloud_base_eff_km']:.0f}–{att['cloud_top_eff_km']:.0f} km"
                 f"  (grey = cloud, light grey = shadow; shading = run-to-run std)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out = os.path.join(SUMMARY_DIR, f"scene_{o['scene']}_am{TAG}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


# ─── cross-scene summary ────────────────────────────────────────────────────
def zone_stats(o):
    rows = []
    x_km, zn, res = o["x_km"], o["zones"], o["res"]
    for surface in o["surfaces"]:
        a, b = res.get(f"{surface}/3d"), res.get(f"{surface}/ipa")
        if a is None or b is None:
            continue
        row = {"scene": o["scene"], "band": o["att"]["band"], "surface": surface,
               "k1_ipa_far": np.nanmean(b["k1_am"][zn["far"]]),
               "k2diag_ipa_far": np.nanmean(b["k2diag_am"][zn["far"]])}
        d = a["k1_am"] - b["k1_am"]
        noise = np.sqrt(np.nanmean(a["k1_am_std"][zn["far"]] ** 2
                                   + b["k1_am_std"][zn["far"]] ** 2) / 3.0)
        row["sig_dk1_far"] = noise
        for z in ("far", "cloud", "shadow", "near"):
            m = zn[z]
            row[f"dk1_{z}"] = np.nanmean(d[m]) if m.any() else np.nan
        im = int(np.nanargmax(np.abs(d)))
        row["dk1_max"], row["x_dk1_max"] = d[im], x_km[im]
        dk2 = a["k2diag_am"] - b["k2diag_am"]
        row["dk2_shadow"] = np.nanmean(dk2[zn["shadow"]]) if zn["shadow"].any() else np.nan
        row["dk2_cloud"] = np.nanmean(dk2[zn["cloud"]])
        im2 = int(np.nanargmax(np.abs(dk2)))
        row["dk2_max"], row["x_dk2_max"] = dk2[im2], x_km[im2]
        da1 = a["a1_am"] - b["a1_am"]
        row["da1_shadow"] = np.nanmean(da1[zn["shadow"]]) if zn["shadow"].any() else np.nan
        rr = a["exp_int_am"] / b["exp_int_am"] - 1
        row["drefl_shadow_pct"] = 100 * np.nanmean(rr[zn["shadow"]]) if zn["shadow"].any() else np.nan
        row["drefl_cloud_pct"] = 100 * np.nanmean(rr[zn["cloud"]])
        pl = (a["plen_mean"][o["iw_mid"]] - b["plen_mean"][o["iw_mid"]]) / 1e3
        row["dplen_shadow_km"] = np.nanmean(pl[zn["shadow"]]) if zn["shadow"].any() else np.nan
        row["dplen_cloud_km"] = np.nanmean(pl[zn["cloud"]])
        row["n_shadow"] = int(zn["shadow"].sum())
        rows.append(row)
    return rows


def write_tables(rows):
    import csv
    cols = list(rows[0].keys())
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(os.path.join(SUMMARY_DIR, f"summary_am{TAG}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    short = ("scene", "band", "surface", "k1_ipa_far", "dk1_far", "sig_dk1_far", "dk1_cloud",
             "dk1_shadow", "dk1_max", "x_dk1_max", "dk2_cloud", "dk2_shadow",
             "drefl_cloud_pct", "drefl_shadow_pct", "dplen_shadow_km", "n_shadow")
    with open(os.path.join(SUMMARY_DIR, f"summary_am{TAG}.md"), "w") as f:
        f.write("| " + " | ".join(short) + " |\n|" + "---|" * len(short) + "\n")
        for r in rows:
            cells = []
            for c in short:
                v = r[c]
                cells.append(v if isinstance(v, str) else (f"{v:d}" if isinstance(v, int)
                             else ("nan" if not np.isfinite(v) else f"{v:.3f}")))
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"Wrote {SUMMARY_DIR}/summary_am{TAG}.csv and .md")


def plot_summary(outs, field, label, tag=""):
    outs = [o for o in outs if o is not None]
    n = len(outs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, o in zip(axes.ravel(), outs):
        x_km, zn, res = o["x_km"], o["zones"], o["res"]
        for surface in o["surfaces"]:
            a, b = res.get(f"{surface}/3d"), res.get(f"{surface}/ipa")
            if a is None or b is None:
                continue
            ax.plot(x_km, a[field] - b[field], color=SURFACE_COLORS.get(surface, "0.3"),
                    lw=1.5, label=surface)
        for m, col in ((zn["cloud"], "0.80"), (zn["shadow"], "0.92")):
            for ix in np.flatnonzero(m):
                ax.axvspan(x_km[ix] - 0.5 * o["att"]["dx_km"], x_km[ix] + 0.5 * o["att"]["dx_km"],
                           color=col, lw=0, zorder=0)
        ax.axhline(0, color="0.4", lw=0.6)
        att = o["att"]
        ax.set_title(f"{o['scene']} ({att['band']}, SZA {att['sza']:.0f}°, VZA {att['sensor_zenith']:.0f}°, "
                     f"COD {att['cloud_cod']:.0f})", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
    fig.suptitle(f"3-D minus IPA, {label}, per scene (grey = cloud, light grey = shadow)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(SUMMARY_DIR, f"summary_delta_{field}{tag}{TAG}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--boxcar-km", type=float, default=0.0,
                    help="top-hat footprint width along x (0 = native 0.5-km columns)")
    a = ap.parse_args()
    global TAG
    TAG = f"_fp{a.boxcar_km:g}" if a.boxcar_km > 0 else ""
    have = sorted(s for s in os.listdir(SCENES_DIR)
                  if os.path.exists(os.path.join(SCENES_DIR, s, "slab_rad.h5")))
    scenes = a.scenes or ([s for s in SCENE_ORDER if s in have]
                          + [s for s in have if s not in SCENE_ORDER])
    outs = [analyse_scene(s, make_fig=not a.no_fig, boxcar_km=a.boxcar_km) for s in scenes]
    outs = [o for o in outs if o is not None]
    rows = sum((zone_stats(o) for o in outs), [])
    write_tables(rows)
    if not a.no_fig:
        # one summary figure per band (the CO2-band ladders are "<band>_<cond>")
        groups = {}
        for o in outs:
            groups.setdefault(o["att"]["band"], []).append(o)
        for band, grp in groups.items():
            tag = "" if band == "o2a" else f"_{band}"
            plot_summary(grp, "k1_am", r"$\Delta k_1^{\rm AM}$", tag)
            plot_summary(grp, "k2diag_am", r"$\Delta k_2^{\rm diag}$", tag)
            plot_summary(grp, "exp_int_am", r"$\Delta e^{c}$", tag)


if __name__ == "__main__":
    main()
