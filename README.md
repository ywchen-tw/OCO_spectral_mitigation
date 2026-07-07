# OCO-2 Cloud-Proximity XCO2 Bias Correction

Pipeline and models for quantifying and correcting cloud-proximity biases in
OCO-2 glint-mode XCO2. Two halves:

1. **Data pipeline** — collocate OCO-2 footprints with Aqua-MODIS cloud masks and
   compute the nearest-cloud distance per sounding (Steps 1–5 below).
2. **Correction stack** — fit photon path-length cumulants (k1/k2) from the L1B
   spectra, build per-sounding feature datasets, train per-surface deep-ensemble
   models that predict the near-cloud XCO2 anomaly, and validate the correction
   independently against TCCON, shipborne EM27/SUN, and ATom aircraft columns.

The correction runs **footprint-by-footprint with no cloud or neighboring-footprint
information at inference** — cloud distance is used only for labels, loss weighting,
and evaluation.

## What the data pipeline does

- **Step 1 (Metadata):** query OCO-2 L1B XML metadata and derive temporal/orbit windows
- **Step 2 (Ingestion):** download OCO-2 products plus MODIS MYD35_L2 and MYD03
- **Step 3 (Processing):** extract OCO-2 footprints, unpack MODIS cloud masks, and match by time
- **Step 4 (Geometry):** convert to ECEF and run KD-Tree distance searches (banded for speed)
- **Step 5 (Integration):** export results and summary statistics

## Quickstart

```bash
cd /Users/yuch8913/programming/oco_fp_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set credentials (required for downloads):

```bash
export EARTHDATA_USERNAME=your_user
export EARTHDATA_PASSWORD=your_pass
export LAADS_TOKEN=your_laads_token
```

Run the full data pipeline in one command:

```bash
python workspace/oco_modis_cloud_distance.py --date 2018-10-18 --visualize --max-distance 20
```

Downstream (per date, after the pipeline): spectral fitting
(`python -m spectral.fitting`), feature dataset
(`src/analysis/build_feature_dataset.py`), model training
(`python -m models.deep_ensemble`), TCCON comparison
(`workspace/tccon_comparison_report.py`). Production configs live in the
`curc_shell_blanca_*.sh` launchers.

## Outputs

- **Pipeline:** `results_YYYY-MM-DD.h5` + stats JSON (see `Config.get_data_path`);
  Step 3/4 intermediates cached under `data/processing/{year}/{doy}/{orbit_id}/`.
- **Spectral fits:** `fitting_details.h5` per date (float32 + gzip).
- **Feature datasets:** `results/csv_collection/combined_<date>_all_orbits.parquet`
  and the multi-year combined parquet.
- **Models + validation:** `results/model_comparison/` — see
  [EXPERIMENTS.md](results/model_comparison/EXPERIMENTS.md) for the experiment
  index; production tag `deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/` holds the
  TCCON / ATom / ship comparison trees.

## Project structure

```
oco_fp_analysis/
├── src/
│   ├── config.py                 # Dataset URLs, paths
│   ├── constants.py              # Single source of pipeline numbers (buffer year,
│   │                             #   anomaly params, band widths, FIT_ORDER)
│   ├── utils.py
│   ├── pipeline/                 # Steps 1–4 (+ step_035_embedding.py, opt-in GEE)
│   ├── spectral/                 # Cumulant fitting: cumulant_fit / orbit_data /
│   │                             #   anomaly / fitting.py facade; FITTING_DERIVATION.md
│   ├── analysis/                 # build_feature_dataset, run_all.py science figures
│   ├── models/                   # deep_ensemble (production), tabm, xgb, gbdt,
│   │                             #   pipeline.py (FeaturePipeline), train_common.py
│   ├── apply/                    # inference bridge (apply_deep_ensemble)
│   ├── search/                   # autoresearch harness (dormant; xgb only)
│   └── abs_util/                 # absorption coefficients
├── workspace/                    # oco_modis_cloud_distance.py, TCCON comparison chain
│   │                             #   (tccon_collocate / ak_harmonize /
│   │                             #    tccon_comparison_report / build_deepens_plot_data)
│   ├── ATom_analysis/            # aircraft pseudo-column ocean validation
│   └── Ship_analysis/            # shipborne EM27/SUN ocean validation
├── prompts/                      # Step specifications and constraints
├── log/                          # PROJECT_REVIEW, plans, notes (archive/ = historical)
├── results/                      # parquets, model comparisons, figures
├── data/
└── requirements.txt
```

## Notes on temporal matching

- Aqua free-drift: `constants.AQUA_FREE_DRIFT_YEAR = 2022`; matching uses ±10 min
  before, ±20 min from that year on. Downloads always use the full ±20 min.

## Key documents

- `log/PROJECT_REVIEW_2026-07-03.md` — scientist/reviewer critique + status of fixes
- `log/NOTES.md` — architecture reference
- `log/CRITICAL_FIXES.md` — pipeline bug ledger
- `src/spectral/FITTING_DERIVATION.md` — k1/k2 physics
- `src/models/deep_ensemble_ARCHITECTURE.md` — production model reference
- `src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md` — uncertainty budget + stats

## Troubleshooting

- If a granule has no cloud pixels, Step 4 will skip distance calculation for that granule.
- Visualization failures are logged as warnings and do not stop the pipeline.

## References

- OCO-2 Data User's Guide: https://docserver.gesdisc.eosdis.nasa.gov/public/project/OCO/OCO2_DUG.pdf
- MODIS Cloud Mask User's Guide: https://modis-atmos.gsfc.nasa.gov/MOD35_L2/
- WGS84 Coordinate System: NIMA TR8350.2
