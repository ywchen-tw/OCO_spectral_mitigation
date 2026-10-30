"""Far-clear cross-band k1 ratio scatter on 29252a_GL: production τ vs Doppler-corrected τ,
single-τ orders 2/3 vs two-τ k1_dry. The ratio SCO2/O2A is the path proxy of the
direct-XCO2 side study; its far-clear scatter (relative) maps 1:1 onto a relative XCO2 error.

Inputs: production fitting details (prod order 7,3,7 on production τ),
results/near_doppler_per_sounding_29252a_GL.parquet (orders 2/3, prod and corrected τ),
results/h2o_per_sounding_29252a_GL.parquet (T2 k1_dry on corrected τ).
Outputs: results/xr_ratio_scatter_29252a_GL.csv, results/xr_ratio_scatter_29252a_GL.png
"""
import warnings
from pathlib import Path
warnings.simplefilter('ignore', RuntimeWarning)
import h5py, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cut_sweep_real import DEFAULTS

OUT = Path(__file__).resolve().parent / "results"
BANDS = ("o2a", "wco2", "sco2")

def robust_sd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))

with h5py.File(DEFAULTS["fit_details"], "r") as f:
    cld, qf, fp, lat = f["cld_dist_km"][...], f["xco2_qf"][...], f["fp_number"][...].astype(int), f["lat"][...]
    prod = {b: f[f"{b}_k1_fitting_nosg"][...] for b in BANDS}
n = len(cld)
far = (cld > 20) & (qf == 0)

k1 = {}  # (label) -> {band: [n]}
k1["prod τ, prod order (7,3,7)"] = prod
nd = pd.read_parquet(OUT / "near_doppler_per_sounding_29252a_GL.parquet")
for o in ("o2", "o3"):
    d = {}
    for b in BANDS:
        s = nd[(nd.src == "prod") & (nd.band == b)].set_index("j")[f"k1_{o}"]
        v = np.full(n, np.nan); v[s.index.values] = s.values; d[b] = v
    k1[f"prod τ, order {o[-1]}"] = d
h2 = pd.read_parquet(OUT / "h2o_per_sounding_29252a_GL.parquet")   # all fits on corrected τ
for col, lab in (("S2_k1", "corrected τ, order 2"), ("S3_k1", "corrected τ, order 3 (k1,k2 ≥ 0)"),
                 ("T2_k1_dry", "corrected τ, two-τ k1_dry (order 2)")):
    d = {}
    for b in BANDS:
        s = h2[h2.band == b].set_index("j")[col]
        v = np.full(n, np.nan); v[s.index.values] = s.values; d[b] = v
    k1[lab] = d

rows = []
series = {}
for lab, d in k1.items():
    for num, den in (("sco2", "o2a"), ("wco2", "o2a"), ("wco2", "sco2")):
        r = d[num] / d[den]
        ok = far & np.isfinite(r)
        s = pd.Series(r[ok]); g = pd.Series(fp[ok])
        rel_fp = (s / s.groupby(g.values).transform("median") - 1).values   # per-footprint medians removed
        rel = (s / s.median() - 1).values
        # along-track averaging: medians of consecutive blocks of 24 far-clear soundings (~3 frames)
        blk = pd.Series(rel_fp).groupby(np.arange(ok.sum()) // 24).median().values
        lag1 = np.corrcoef(rel_fp[:-1], rel_fp[1:])[0, 1]
        rows.append(dict(fit=lab, ratio=f"{num}/{den}", n_far=int(ok.sum()), median=s.median(),
                         spread_pct=100 * robust_sd(rel), spread_fp_pct=100 * robust_sd(rel_fp),
                         spread_block24_pct=100 * robust_sd(blk), lag1=lag1,
                         ppm_equiv=100 * robust_sd(rel_fp) * 4.1))
        if (num, den) == ("sco2", "o2a"):
            series[lab] = (lat[ok], rel_fp)
    for b in BANDS:
        v = d[b]; ok = far & np.isfinite(v); s = pd.Series(v[ok]); g = pd.Series(fp[ok])
        rows.append(dict(fit=lab, ratio=b, n_far=int(ok.sum()), median=s.median(),
                         spread_pct=100 * robust_sd(s / s.median() - 1),
                         spread_fp_pct=100 * robust_sd(s / s.groupby(g.values).transform("median") - 1),
                         spread_block24_pct=np.nan, lag1=np.nan, ppm_equiv=np.nan))
df = pd.DataFrame(rows)
df.to_csv(OUT / "xr_ratio_scatter_29252a_GL.csv", index=False)
pd.set_option("display.width", 200)
print(df[df.ratio.str.contains("/")].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print()
print(df[~df.ratio.str.contains("/")].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

fig, axes = plt.subplots(len(series), 1, figsize=(10, 1.9 * len(series)), sharex=True)
for ax, (lab, (la, rel)) in zip(axes, series.items()):
    ax.plot(la, 100 * rel, ".", ms=2, color="0.3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-15, 15)
    ax.set_ylabel("%")
    ax.text(0.01, 0.85, f"{lab}: robust SD {100 * robust_sd(rel):.2f} %", transform=ax.transAxes, fontsize=9)
axes[-1].set_xlabel("latitude (far-clear soundings, cld > 20 km, QF0)")
fig.suptitle("29252a_GL: SCO2/O2A k1 ratio, per-footprint medians removed", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "xr_ratio_scatter_29252a_GL.png", dpi=130)
