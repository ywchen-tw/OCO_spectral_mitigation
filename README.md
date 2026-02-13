# OCO-2/MODIS Footprint Analysis

A step-based workflow for collocating OCO-2 glint-mode footprints with Aqua MODIS cloud masks and computing the nearest-cloud distance for each OCO-2 sounding.

## What this project does

- **Step 1 (Metadata):** query OCO-2 L1B XML metadata and derive temporal/orbit windows
- **Step 2 (Ingestion):** download OCO-2 products plus MODIS MYD35_L2 and MYD03
- **Step 3 (Processing):** extract OCO-2 footprints, unpack MODIS cloud masks, and match by time
- **Step 4 (Geometry):** convert to ECEF and run KD-Tree distance searches (banded for speed)
- **Step 5 (Integration):** export results and summary statistics

## Quickstart

```bash
cd /Users/yuch8913/programming/oco/fp_analysis
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

Run the full pipeline in one command:

```bash
python workspace/demo_combined.py --date 2018-10-18 --visualize --max-distance 20
```

## Outputs

By default, results are written to the configured data directory (see `Config.get_data_path`).

- **HDF5:** `results_YYYY-MM-DD.h5`
- **Stats JSON:** `results_YYYY-MM-DD.json`
- **Visualizations (optional):** `visualizations_combined/` and per-granule subfolders

CSV output is intentionally disabled.

## Cache layout

Step 3 and Step 4 cache their intermediate outputs under:

```
data/processing/{year}/{doy}/{orbit_id}/
  myd35_*.pkl
  granule_combined_*.pkl
  phase4_results.pkl
```

This lets you re-run the pipeline quickly without reprocessing all granules.

## Project structure

```
fp_analysis/
├── src/
│   ├── config.py                # Dataset URLs, constants, paths
│   ├── utils.py                 # Common helpers
│   ├── phase_01_metadata.py     # XML metadata parsing
│   ├── phase_02_ingestion.py    # Download management
│   ├── phase_03_processing.py   # Footprints + cloud masks
│   └── phase_04_geometry.py     # Distance calculations + visualizations
├── workspace/
│   ├── demo_combined.py         # End-to-end pipeline
│   ├── demo_phase_01.py
│   ├── demo_phase_02.py
│   ├── demo_phase_03.py
│   └── demo_phase_04.py
├── prompts/                      # Step specifications and constraints
├── log/
├── data/
└── requirements.txt
```

## Notes on temporal matching

- Aqua is in free-drift (2023+), so the matching window must be wider.
- `Config.MODIS_TEMPORAL_BUFFER_MINUTES` is set to 20 minutes.
- `workspace/demo_combined.py` currently uses a 30-minute matching window for MODIS-to-OCO-2 assignment.

## Prompts

The `prompts/` folder documents the intended step-by-step behavior and scientific constraints:

- `OCO2_data_analyst.md`
- `Phase_01_Metadata.md`
- `Phase_02_Ingestion.md`
- `Phase_03_Processing.md`
- `Phase_04_Geometry.md`
- `Phase_05_Synthesis.md`

## Troubleshooting

- If a granule has no cloud pixels, Step 4 will skip distance calculation for that granule.
- Visualization failures are logged as warnings and do not stop the pipeline.

## References

- OCO-2 Data User's Guide: https://docserver.gesdisc.eosdis.nasa.gov/public/project/OCO/OCO2_DUG.pdf
- MODIS Cloud Mask User's Guide: https://modis-atmos.gsfc.nasa.gov/MOD35_L2/
- WGS84 Coordinate System: NIMA TR8350.2
