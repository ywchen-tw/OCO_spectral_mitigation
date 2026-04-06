"""
pca_analyze_reduced.py  —  ENTRY POINT
========================================
PCA analysis using only k3, albedo, and ancillary variables.

Excluded vs pca_analyze.py:
  - k1/k2/k3 for all bands  (o2a/wco2/sco2 × k1/k2/k3)
  - exp_intercept for all bands  (exp_o2a/wco2/sco2_intercept)
  - exp−albedo differences  (o2a_exp_intercept-alb, wco2_exp_intercept-alb)

Retained features (all non-k, non-exp pipeline features):
  - albedo:    alb_o2a, alb_wco2, alb_sco2
  - geometry:  cos_glint_angle, 1_over_cos_sza, 1_over_cos_vza, sin_raa
  - atmosphere: log_P, dp, dp_psfc_prior_ratio, h2o_scale, delT, co2_grad_del,
               co2_ratio_bc, h2o_ratio_bc
  - signal:    csnr_o2a, csnr_sco2, h_cont_o2a, h_cont_sco2
  - aerosol:   aod_dust, aod_oc, aod_seasalt, aod_strataer, aod_sulfate
  - ancillary: fp_area_km2, pol_ang_rad, s31, tcwv

Input
-----
- combined_2020-01-01_all_orbits.parquet  (macOS)
- combined_2016_2020_dates.parquet        (Linux / full dataset)

Output
------
results/figures/cld_dist_analysis/pca_reduced/
    pca_scree.png
    pca_loadings_heatmap.png
    pca_scores_vs_cld_dist.png
    pca_biplot.png
    pca_correlation_with_cld_dist.png
    pca_scores_vs_xco2.png
    pca_scores_by_cld_dist_group.png
results/figures/cld_dist_analysis/ocean/pca_reduced/   (same, tag=ocean)
results/figures/cld_dist_analysis/land/pca_reduced/    (same, tag=land)
"""

import gc
import sys
import logging
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

from ca_utils import (
    get_storage_dir, load_data, apply_quality_filter,
    cld_dist_bins,
)
from ca_pca import run_pca_analysis

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── reduced feature set: no k1/k2/k3 or exp_intercept ────────────────────────
_REDUCED_FEATURES = [
    # albedo
    'alb_o2a', 'alb_wco2', 'alb_sco2',
    # geometry / viewing angles
    'cos_glint_angle',
    '1_over_cos_sza', '1_over_cos_vza', 'sin_raa',
    # atmosphere / retrieval state
    'log_P', 'dp', 'dp_psfc_prior_ratio',
    'h2o_scale', 'delT', 'co2_grad_del',
    'co2_ratio_bc', 'h2o_ratio_bc',
    # signal quality
    'csnr_o2a', 'csnr_sco2',
    'h_cont_o2a', 'h_cont_sco2',
    # aerosol components
    'aod_dust', 'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
    # ancillary
    'fp_area_km2', 'pol_ang_rad', 's31', 'tcwv',
]


def main():
    storage_dir = get_storage_dir()
    result_dir  = storage_dir / 'results'
    csv_dir     = result_dir / 'csv_collection'

    # ── load ──────────────────────────────────────────────────────────────────
    if platform.system() == 'Darwin':
        df = load_data(csv_dir, parquet_fname='combined_2020-01-01_all_orbits.parquet')
    elif platform.system() == 'Linux':
        df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')
    else:
        df = load_data(csv_dir)

    # ── quality filter ────────────────────────────────────────────────────────
    df = apply_quality_filter(df)

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    base_outdir = result_dir / 'figures' / 'cld_dist_analysis'

    # ── combined (all surfaces) ───────────────────────────────────────────────
    logger.info("Running reduced PCA on combined (all surfaces) …")
    run_pca_analysis(df, bins, labels,
                     outdir=str(base_outdir / 'pca_reduced'),
                     tag='',
                     features=_REDUCED_FEATURES)
    gc.collect()

    # ── per surface type ──────────────────────────────────────────────────────
    sfc_codes = {'ocean': 0, 'land': 1} if 'sfc_type' in df.columns else {}

    for sfc_name, sfc_code in sfc_codes.items():
        sdf = df[df['sfc_type'] == sfc_code]
        logger.info(f"\n{'='*55}\nReduced PCA for surface type: {sfc_name.upper()}  "
                    f"({len(sdf):,} soundings)\n{'='*55}")
        outdir = str(base_outdir / sfc_name / 'pca_reduced')
        run_pca_analysis(sdf, bins, labels,
                         outdir=outdir,
                         tag=sfc_name,
                         features=_REDUCED_FEATURES)
        del sdf
        gc.collect()


if __name__ == '__main__':
    main()
