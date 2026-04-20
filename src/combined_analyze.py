"""
combined_analyze.py  —  ENTRY POINT ONLY
=========================================
Run all analysis sections.  All plot functions live in the ca_* modules:

  ca_utils.py          shared helpers (I/O, binning, rolling stats)
  ca_signal.py         Section 1: signal hierarchy
  ca_exp_alb.py        Section 2: exp_intercept / albedo analyses
  ca_k_coeff.py        Section 3: k1 / k2 / k3 analyses
  ca_stratified.py     Section 4: stratified analyses
  ca_xco2.py           Section 5: XCO2 anomaly analyses
  ca_ref_corrected.py  Sections R0–R7: ref-corrected analyses

Original combined_analyze.py docstring preserved below for reference.
----------------------------------------------------------------------
Analyze parquet output files from fitting_data_correction.py.
Collocates OCO-2 footprint spectral coefficients (k1/k2/k3, exp_intercept,
albedo) with cloud-proximity distance (cld_dist_km) and produces diagnostic
figures for ocean and land separately.

Input
-----
- combined_2016_2020_dates.parquet  (or per-date combined_*_all_orbits.parquet)
  written by fitting_data_correction.py::raw_processing_single_date()

Output
------
results/figures/cld_dist_analysis/
    signal_hierarchy.png
    residual_signal_hierarchy.png
    exp_intercept_albedo_residuals.png
    alb_exp_pct_change_vs_cld_dist.png
    alb_exp_ratio_divergence_vs_cld_dist.png
    exp_alb_ratio_residuals.png
    k{1,2,3}_albedo_residuals.png
    exp_intercept_interband_coherence.png
    higher_order_k_profiles.png
    ocean/  land/
        dist_vs_cld_dist_boxplot.png
        exp_intercept_binned_profile.png
        alb_vs_exp_intercept.png
        alb_vs_exp_intercept_cross.png
        k1_k2_binned_profile.png
        k1_k2_vs_cld_dist.png
        k2_over_k1_vs_cld_dist.png
        k1_vs_k2_joint_cld_dist.png
        cross_band_k_ratio_profiles.png
        cross_band_k1_scatter_matrix.png
        cross_band_k2_scatter_matrix.png
        cross_band_k3_scatter_matrix.png
        alb_binned_profile.png
        xco2_anomaly_partial_vs_cld_dist.png
        xco2_bc_anomaly_vs_predictors.png
        xco2_{bc,raw}_anomaly_vs_cld_dist_binned.png
        xco2_anomaly_correlation_heatmap.png
        xco2_raw_minus_apriori_vs_cld_dist_binned.png
        xco2_raw_minus_strong_idp_vs_cld_dist_binned.png
        xco2_raw_minus_apriori_vs_bc_anomaly.png
        xco2_raw_minus_strong_idp_vs_bc_anomaly.png
        stratified/by_{var}/{bin}/  (Section 4)
    ref_corrected/   (Sections R0–R7, requires ref_* columns)
        ref_diff_scatter_{k1,k2,alb,exp}.png
        ref_coverage_bias.png
        ref_std_profiles.png
        ref_corrected_{k1,k2,alb,exp}_profiles.png
        ref_zscore_{k1,k2,alb,exp}_profiles.png
        ref_signal_hierarchy.png
        ref_alb_decoupled_exp_residuals.png
        obs_vs_ref_scatter_{ocean,land}.png
    r25_corrected/   (Sections R0–R7 with r25 reference, requires r25_* columns)
        r25_diff_scatter_{k1,k2,alb,exp}.png
        r25_coverage_bias.png
        r25_std_profiles.png
        r25_corrected_{k1,k2,alb,exp}_profiles.png
        r25_zscore_{k1,k2,alb,exp}_profiles.png
        r25_signal_hierarchy.png
        r25_alb_decoupled_exp_residuals.png
        obs_vs_r25_scatter_{ocean,land}.png

Code structure
--------------
Helpers
    get_storage_dir()           Platform-aware data root (macOS / Linux / default)
    load_data()                 Load combined parquet or concatenate per-date parquets
    apply_quality_filter()      xco2_bc > 0, xco2_qf == 0, snow_flag == 0; float32 downcast
    split_by_surface()          sfc_type 0=ocean / 1=land
    cld_dist_bins()             Fixed cloud-distance bin edges + labels
    bin_by_cld_dist()           pd.cut wrapper
    rolling_median_iqr()        O(n) binned rolling median + IQR (replaces O(n²) window)

Section 1  Signal hierarchy
    plot_signal_hierarchy()             Pearson r(cld_dist) bar chart — k1/k2/k3/exp/exp-alb
    plot_residual_signal_hierarchy()    Same after OLS-removing albedo + airmass + cos(SZA)

Section 2  exp_intercept / albedo analyses
    plot_alb_vs_exp_intercept()         Within-band scatter + rolling median
    plot_alb_vs_exp_intercept_cross()   3×3 cross-band scatter matrix
    plot_intercept_binned_profile()     Binned mean ± SEM/std vs cld_dist
    plot_exp_intercept_interband_coherence()  Pairwise scatter colored by cld_dist
    plot_alb_exp_divergence()           % change from far-cloud ref; exp/alb ratio divergence
    plot_exp_intercept_albedo_residuals()     OLS residuals after alb+airmass+SZA removal
    plot_exp_alb_ratio_residuals()      OLS residuals of exp/alb after airmass+SZA+AOD removal

Section 3  k1 / k2 / k3 analyses
    plot_k1_k2_binned_profile()         Binned mean ± SEM/std for k1 and k2
    plot_k1_k2_vs_cld_dist()           Hexbin scatter + rolling median
    plot_k2_over_k1_vs_cld_dist()      k2/k1 scattering asymmetry ratio
    plot_k1_k2_joint()                 k1 vs k2 scatter colored by cld_dist
    plot_higher_order_k_profiles()      k3 binned profiles for SCO2 and WCO2
    plot_k_albedo_residuals()           OLS residuals of k1/k2/k3 after alb+airmass+SZA
    plot_cross_band_k_combinations()    Cross-band kN ratios (binned profiles) + scatter matrix per k-order

Section 4  Stratified analyses
    STRAT_CONFIG                        Dict of stratification variables and bin edges
    _build_strata()                     Clip edges to data range; assign _strat column
    run_stratified_analysis()           Per-stratum core plots + overlay comparisons
    plot_k1_k2_overlay()                All strata k1/k2 profiles on one figure
    plot_intercept_overlay()            All strata exp_intercept profiles on one figure
    plot_xco2_anomaly_binned_overlay()  All strata XCO2 anomaly profiles on one figure

Section 5  XCO2 anomaly
    plot_xco2_anomaly_correlations()    Pearson r heat-map vs all key predictors
    plot_xco2_anomaly_vs_key_vars()     Scatter panels vs top predictors
    plot_xco2_anomaly_vs_cld_dist_binned()  Mean ± SEM bar chart by cld_dist bin
    plot_xco2_anomaly_partial()         Partial correlation after OLS-removing confounders

Supplementary
    plot_distributions_vs_cld_dist()    Box-plots of key variables by cld_dist bin
    plot_alb_binned_profile()           Albedo binned profiles

Ref-corrected analyses (R0–R7)   — requires ref_* or r25_* columns
    Registry
        _REF_PAIRS          (obs, ref_mean, ref_std, diff_col, band, term, color) × 12
        _R25_PAIRS          Same structure for r25_* reference (min_cld_dist=25 km)
    Helpers
        add_ref_anomalies() / add_r25_anomalies()   Compute obs-ref diff and z-score columns
        _has_ref_data()     / _has_r25_data()        Presence checks
        _binned_ref_profile()                        Shared binned-profile subplot helper
    Plots (all accept pairs=, tag= for ref / r25 dispatch)
        R0  plot_ref_diff_vs_cld_dist()              Hexbin scatter of obs−ref vs cld_dist
        R1  plot_ref_coverage_bias()                 Selection bias: has-ref vs no-ref
        R2  plot_ref_std_profiles()                  Reference σ vs cld_dist
        R3  plot_ref_corrected_profiles()            Binned mean ± SEM of obs−ref
        R4  plot_ref_zscore_profiles()               Binned mean ± SEM of (obs−ref)/σ_ref
        R5  plot_ref_signal_hierarchy()              r(cld_dist, obs−ref) bar chart
        R6  plot_ref_alb_decoupled_exp()             OLS-remove Δalb from Δexp residuals
        R7  plot_obs_vs_ref_scatter()                Hexbin obs vs ref_mean with 1:1 line

Entry point
    main()   Load → QF filter → cloud-dist bins → run all sections → per-surface loop
"""

import gc
import sys
import logging
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── module imports ─────────────────────────────────────────────────────────────
from ca_utils import (
    get_storage_dir, load_data, apply_quality_filter,
    cld_dist_bins, bin_by_cld_dist, print_summary_stats,
)
from ca_signal import (
    plot_signal_hierarchy, plot_residual_signal_hierarchy,
)
from ca_exp_alb import (
    plot_alb_vs_exp_intercept, plot_alb_vs_exp_intercept_cross,
    plot_intercept_binned_profile, plot_exp_intercept_interband_coherence,
    plot_alb_exp_divergence, plot_exp_intercept_albedo_residuals,
    plot_exp_alb_ratio_residuals, plot_alb_binned_profile,
)
from ca_k_coeff import (
    plot_distributions_vs_cld_dist, plot_k1_k2_vs_cld_dist,
    plot_k2_over_k1_vs_cld_dist, plot_k1_k2_binned_profile,
    plot_k1_k2_joint, plot_higher_order_k_profiles,
    plot_k_albedo_residuals, plot_cross_band_k_combinations,
    plot_fp_area_analysis, plot_xco2_anomaly_ocean_land,
)
from ca_stratified import STRAT_CONFIG, run_stratified_analysis
from ca_xco2 import (
    plot_xco2_derived_vs_cld_dist_binned, plot_xco2_derived_vs_bc_anomaly,
    plot_xco2_anomaly_vs_cld_dist_binned,
    run_xco2_sign_analysis,
    _XCO2_TARGET_CONFIG, run_xco2_target_analysis,
)
from ca_ref_corrected import (
    _REF_PAIRS, _R25_PAIRS,
    _has_ref_data, _has_r25_data,
    add_ref_anomalies, add_r25_anomalies,
    plot_ref_diff_vs_cld_dist, plot_ref_coverage_bias,
    plot_ref_std_profiles, plot_ref_corrected_profiles,
    plot_ref_zscore_profiles, plot_ref_signal_hierarchy,
    plot_ref_alb_decoupled_exp, plot_obs_vs_ref_scatter,
    # R8–R12
    plot_ref_delta_multivar, plot_ref_cross_band_delta,
    plot_ref_delta_decay, plot_ref_delta_vs_xco2,
    plot_ref_delta_partial_xco2,
    # R14
    plot_ref_corrected_profiles_by_fp_area,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _run_subset_analysis(sdf, bins, labels, subset_name, subset_outdir):
    logger.info(f"\n{'='*55}\nRunning analysis for subset: {subset_name.upper()}\n{'='*55}")
    logger.info(f"  {subset_name} soundings: {len(sdf):,}")

    print_summary_stats(sdf, bins, labels)

    # ── Section 1 (per subset): distributions ───────────────────────────────
    logger.info("Plotting distributions vs cloud distance …")
    plot_distributions_vs_cld_dist(sdf, bins, labels, subset_outdir)

    # ── Section 2c: exp_intercept binned profiles ───────────────────────────
    logger.info("Plotting intercept binned profiles …")
    plot_intercept_binned_profile(sdf, bins, labels, subset_outdir)

    # ── Section 2a: albedo vs exp_intercept ─────────────────────────────────
    logger.info("Plotting albedo vs exp_intercept …")
    plot_alb_vs_exp_intercept(sdf, subset_outdir)

    logger.info("Plotting albedo vs exp_intercept cross-band …")
    plot_alb_vs_exp_intercept_cross(sdf, subset_outdir)

    # ── Section 3a: k1/k2 profiles and scatter ──────────────────────────────
    logger.info("Plotting k1/k2 binned profiles …")
    plot_k1_k2_binned_profile(sdf, bins, labels, subset_outdir)

    logger.info("Plotting k1/k2 scatter vs cloud distance …")
    plot_k1_k2_vs_cld_dist(sdf, subset_outdir)

    # ── Section 3b: k2/k1 ratio ──────────────────────────────────────────────
    logger.info("Plotting k2/k1 ratio vs cloud distance …")
    plot_k2_over_k1_vs_cld_dist(sdf, subset_outdir)

    # ── Section 3d: k1 vs k2 joint scatter ──────────────────────────────────
    logger.info("Plotting k1 vs k2 joint colored by cld_dist …")
    plot_k1_k2_joint(sdf, subset_outdir)

    # ── Section 3f: cross-band k combinations ───────────────────────────────
    logger.info("Plotting cross-band k combination profiles and scatter matrix …")
    _k_cols = ['cld_dist_km'] + [c for c in sdf.columns
                                 if c.rsplit('_', 1)[-1] in ('k1', 'k2', 'k3')]
    plot_cross_band_k_combinations(sdf[_k_cols].copy(), bins, labels, subset_outdir)
    del _k_cols

    # ── R13: footprint area analysis ─────────────────────────────────────────
    logger.info("R13: Footprint area vs spectral variables …")
    plot_fp_area_analysis(sdf, bins, labels, subset_outdir)

    # ── Supplementary ────────────────────────────────────────────────────────
    logger.info("Plotting albedo binned profiles …")
    plot_alb_binned_profile(sdf, bins, labels, subset_outdir)

    # ── Section 5: XCO2 binned profiles directly in subset_outdir ───────────
    logger.info("Plotting XCO2 binned profiles (all targets) …")
    plot_xco2_anomaly_vs_cld_dist_binned(
        sdf, bins, labels, subset_outdir,
        targets=[(col, lbl, clr)
                 for (col, lbl, _), clr in zip(_XCO2_TARGET_CONFIG,
                                               ['C0', 'C1', 'C2', 'C3'])
                 if col in sdf.columns])
    gc.collect()

    # ── Section 5: XCO2 full suite (one subfolder per target) ───────────────
    xco2_base = str(Path(subset_outdir) / 'xco2')
    for _col, _lbl, _ in _XCO2_TARGET_CONFIG:
        logger.info(f"Section 5 [{_col}]: running full XCO2 plot suite …")
        run_xco2_target_analysis(sdf, bins, labels,
                                 xco2_base, _col, _lbl)
        gc.collect()

    # ── Section 5b: xco2_raw_minus_apriori / strong_idp analyses ────────────
    logger.info("Plotting xco2 derived quantities vs cld_dist binned …")
    plot_xco2_derived_vs_cld_dist_binned(sdf, bins, labels, subset_outdir)

    for _col, _lbl, _ in _XCO2_TARGET_CONFIG:
        logger.info(f"Plotting xco2 derived quantities vs {_col} …")
        plot_xco2_derived_vs_bc_anomaly(sdf, bins, labels, subset_outdir,
                                         y_col=_col, y_label=_lbl)
        gc.collect()

    logger.info(f"All figures for {subset_name} written to {subset_outdir}")

    # ── Part 3: XCO2 sign-split analyses ─────────────────────────────────────
    for _col, _lbl, _ in _XCO2_TARGET_CONFIG:
        logger.info(f"Running XCO2 sign-split analysis [{_col}] for {subset_name.upper()} …")
        run_xco2_sign_analysis(sdf, bins, labels, subset_outdir,
                               run_ref=_has_ref_data(sdf),
                               split_col=_col, split_label=_lbl)
        gc.collect()

    # ── Section 4: stratified analyses ───────────────────────────────────────
    logger.info(f"Running stratified analyses for {subset_name.upper()} …")
    for strat_var, (strat_edges, strat_unit) in STRAT_CONFIG.items():
        if strat_var not in sdf.columns:
            continue
        run_stratified_analysis(sdf, bins, labels, subset_outdir,
                                strat_var, strat_edges, strat_unit)


def _subset_for_fp(df, fp_idx):
    fp_col = f'fp_{fp_idx}'
    if fp_col in df.columns:
        return df[df[fp_col] == 1]

    if 'fp_number' in df.columns:
        fp_vals = df['fp_number'].dropna().astype(int)
        if fp_vals.empty:
            return df.iloc[0:0]

        unique_vals = set(fp_vals.unique().tolist())
        if set(range(8)).issubset(unique_vals) or (unique_vals and min(unique_vals) == 0):
            return df[df['fp_number'].astype(int) == fp_idx]

        if set(range(1, 9)).issubset(unique_vals) or (unique_vals and min(unique_vals) == 1):
            return df[df['fp_number'].astype(int) == (fp_idx + 1)]

        return df[df['fp_number'].astype(int) == fp_idx]

    logger.warning("Neither fp_number nor fp_0..fp_7 columns found — cannot split by footprint")
    return None



# ── main ──────────────────────────────────────────────────────────────────────

def main():
    storage_dir = get_storage_dir()
    result_dir  = storage_dir / 'results'
    csv_dir     = result_dir / 'csv_collection'
    # ── load ──────────────────────────────────────────────────────────────────
    if platform.system() == 'Darwin':
        df = load_data(csv_dir, parquet_fname='combined_2020-01-01_all_orbits.parquet')
        # df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')
    elif platform.system() == 'Linux':
        df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')
    else:
        df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')

    # ── quality filter (snow excluded, surface split done below) ──────────────
    df = apply_quality_filter(df)

    # ── scale exp_intercept by π ──────────────────────────────────────────────
    # # TODO: remove this once the π factor is absorbed into oco_fp_spec_anal.py
    # # Scale both obs and ref exp_int so they stay on the same scale for diffs.
    # _exp_int_cols = [
    #     'exp_o2a_intercept',    'exp_wco2_intercept',    'exp_sco2_intercept',
    #     'ref_exp_int_o2a_mean', 'ref_exp_int_wco2_mean', 'ref_exp_int_sco2_mean',
    #     'ref_exp_int_o2a_std',  'ref_exp_int_wco2_std',  'ref_exp_int_sco2_std',
    # ]
    # for _col in _exp_int_cols:
    #     if _col in df.columns:
    #         df[_col] = df[_col]

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    # ── Section 1: signal hierarchy (full df, internal ocean/land split) ─────
    overall_outdir = str(result_dir / 'figures' / 'cld_dist_analysis')
    logger.info("Section 1: Plotting signal hierarchy (r vs cld_dist) …")
    plot_signal_hierarchy(df, overall_outdir)

    # ── Section 1b: residual signal hierarchy ─────────────────────────────────
    logger.info("Section 1b: Plotting residual signal hierarchy …")
    plot_residual_signal_hierarchy(df, overall_outdir)

    # ── Section 2b: exp_intercept albedo residuals ────────────────────────────
    logger.info("Section 2b: Plotting exp_intercept albedo residuals …")
    plot_exp_intercept_albedo_residuals(df, bins, labels, overall_outdir)

    # ── Section 2e: albedo vs exp_intercept divergence ───────────────────────
    logger.info("Section 2e: Plotting alb vs exp_intercept divergence …")
    plot_alb_exp_divergence(df, bins, labels, overall_outdir)

    # ── Section 2f: exp/alb ratio residuals ──────────────────────────────────
    logger.info("Section 2f: Plotting exp/alb ratio residuals …")
    plot_exp_alb_ratio_residuals(df, bins, labels, overall_outdir)

    # ── Section 3e: k1, k2, k3 albedo residuals ──────────────────────────────
    logger.info("Section 3e: Plotting k1/k2/k3 albedo residuals …")
    plot_k_albedo_residuals(df, bins, labels, overall_outdir)

    # ── Section 2d: exp_intercept inter-band coherence ────────────────────────
    logger.info("Section 2d: Plotting exp_intercept inter-band coherence …")
    plot_exp_intercept_interband_coherence(df, overall_outdir)

    # ── Section 3c: higher-order k profiles (k3 for SCO₂ and WCO₂) ──────────
    logger.info("Section 3c: Plotting higher-order k3 profiles …")
    plot_higher_order_k_profiles(df, bins, labels, overall_outdir)

    # ── Sections R1–R7: ref-corrected analyses ────────────────────────────────
    if _has_ref_data(df):
        ref_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / 'ref_corrected')
        logger.info("Adding ref-corrected anomaly columns …")
        df_r = add_ref_anomalies(df)

        logger.info("R0: fp − ref scatter vs cloud distance …")
        plot_ref_diff_vs_cld_dist(df_r, ref_outdir)

        logger.info("R1: Ref coverage bias analysis …")
        plot_ref_coverage_bias(df_r, bins, labels, ref_outdir)

        logger.info("R2: Ref std profiles (scene heterogeneity) …")
        plot_ref_std_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R3: Ref-corrected anomaly profiles …")
        plot_ref_corrected_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R4: Ref z-score profiles …")
        plot_ref_zscore_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R5: Ref-corrected signal hierarchy …")
        plot_ref_signal_hierarchy(df_r, ref_outdir)

        logger.info("R6: Ref albedo-decoupled exp_intercept residuals …")
        plot_ref_alb_decoupled_exp(df_r, bins, labels, ref_outdir)

        logger.info("R7: Obs vs ref scatter …")
        plot_obs_vs_ref_scatter(df_r, ref_outdir)

        logger.info("R8: Multi-variable delta comparison …")
        plot_ref_delta_multivar(df_r, bins, labels, ref_outdir)

        logger.info("R9: Cross-band delta coherence …")
        plot_ref_cross_band_delta(df_r, ref_outdir)

        logger.info("R10: Delta decay length scale …")
        plot_ref_delta_decay(df_r, bins, labels, ref_outdir)

        logger.info("R11: Delta vs XCO2 BC anomaly …")
        plot_ref_delta_vs_xco2(df_r, ref_outdir)

        logger.info("R12: Partial correlation of delta vs XCO2 anomaly …")
        plot_ref_delta_partial_xco2(df_r, ref_outdir)

        logger.info("R14: Ref-corrected profiles by footprint area quintile …")
        plot_ref_corrected_profiles_by_fp_area(df_r, bins, labels, ref_outdir)

        logger.info(f"All ref-corrected figures written to {ref_outdir}")
        del df_r
        gc.collect()
    else:
        logger.warning("No ref_* columns found — skipping Sections R1–R7")

    # ── Sections R1–R7 (r25 reference, min_cld_dist=25 km) ───────────────────
    if 0:#_has_r25_data(df):
        r25_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / 'r25_corrected')
        logger.info("Adding r25-corrected anomaly columns …")
        df_r25 = add_r25_anomalies(df)

        logger.info("R0 [r25]: fp − r25 scatter vs cloud distance …")
        plot_ref_diff_vs_cld_dist(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R1 [r25]: r25 coverage bias analysis …")
        plot_ref_coverage_bias(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R2 [r25]: r25 std profiles (scene heterogeneity) …")
        plot_ref_std_profiles(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R3 [r25]: r25-corrected anomaly profiles …")
        plot_ref_corrected_profiles(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R4 [r25]: r25 z-score profiles …")
        plot_ref_zscore_profiles(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R5 [r25]: r25-corrected signal hierarchy …")
        plot_ref_signal_hierarchy(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R6 [r25]: r25 albedo-decoupled exp_intercept residuals …")
        plot_ref_alb_decoupled_exp(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R7 [r25]: Obs vs r25 scatter …")
        plot_obs_vs_ref_scatter(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R8 [r25]: Multi-variable delta comparison …")
        plot_ref_delta_multivar(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R9 [r25]: Cross-band delta coherence …")
        plot_ref_cross_band_delta(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R10 [r25]: Delta decay length scale …")
        plot_ref_delta_decay(df_r25, bins, labels, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R11 [r25]: Delta vs XCO2 BC anomaly …")
        plot_ref_delta_vs_xco2(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info("R12 [r25]: Partial correlation of delta vs XCO2 anomaly …")
        plot_ref_delta_partial_xco2(df_r25, r25_outdir, pairs=_R25_PAIRS, tag='r25')

        logger.info(f"All r25-corrected figures written to {r25_outdir}")
        del df_r25
        gc.collect()
    else:
        logger.warning("No r25_* columns found — skipping r25 Sections R1–R7")

    # ── surface-type loop: process ocean then land sequentially ───────────────
    sfc_codes = {'ocean': 0, 'land': 1} if 'sfc_type' in df.columns else {'all': None}

    for sfc_name, sfc_code in sfc_codes.items():
        sdf = df[df['sfc_type'] == sfc_code] if sfc_code is not None else df
        sfc_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / sfc_name)
        _run_subset_analysis(sdf, bins, labels, sfc_name, sfc_outdir)
        del sdf
        gc.collect()

    # ── footprint loop: fp_0 .. fp_7 (same suite as per-surface) ────────────
    for fp_idx in range(8):
        fp_name = f'fp_{fp_idx}'
        fp_df = _subset_for_fp(df, fp_idx)
        if fp_df is None:
            break
        if fp_df.empty:
            logger.warning(f"No rows for {fp_name} — skipping")
            continue

        fp_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / 'footprints' / fp_name)
        _run_subset_analysis(fp_df, bins, labels, fp_name, fp_outdir)
        del fp_df
        gc.collect()

    # ── Ocean vs Land XCO2 boxplots for all targets (uses full df) ───────────
    combined_outdir = str(result_dir / 'figures' / 'cld_dist_analysis')
    for _col, _lbl, _ in _XCO2_TARGET_CONFIG:
        logger.info(f"Plotting {_col} ocean vs land boxplot …")
        plot_xco2_anomaly_ocean_land(df, bins, labels, combined_outdir,
                                     col=_col, label=_lbl)
        gc.collect()


if __name__ == '__main__':
    main()
