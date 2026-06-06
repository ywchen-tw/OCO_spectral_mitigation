# CMR and GES DISC Download Notes

Date recorded: 2026-06-06

## Context

The Phase 2 OCO-2 ingestion occasionally reported "no data" or failed later in
Phase 3 even when the data existed on GES DISC. Manual CURC checks showed that
plain GES DISC `/data/OCO2_DATA/...` URLs were reachable, while browser-derived
tokenized `/data/.TOKEN/...` URLs could be stale or invalid.

Example observations from CURC:

- Plain L1B directory returned `200 OK`.
- Tokenized `.TOKEN` directory returned `404 Not Found`.
- Plain L1B file URL returned `200 OK`.
- Directory HTML could contain Earthdata/URS references even when it was not an
  actual login page.

## Main Issues Found

1. Stale `GESDISC_DATA_TOKEN`

   Browser-derived `.TOKEN` paths are context-specific and can stop working in
   batch jobs. Using that token in Phase 2 made otherwise valid GES DISC paths
   fail before the plain `/data/OCO2_DATA/...` route was tried.

2. Over-broad auth-page detection

   The downloader previously treated ordinary GES DISC listing pages as login
   pages if they merely contained Earthdata/URS-related text. This caused false
   auth failures and prevented useful fallback behavior.

3. Fragile CMR queries

   L1B CMR lookup needed the base short name plus version, not only stale
   versioned short-name assumptions. Atom metadata also needed support for
   `time:start`, `time:end`, and `echo:producerGranuleId` fields.

4. Secondary product fallback gaps

   Lite, Met, and CO2Prior downloads depended heavily on GES DISC directory
   listings. When those listings returned a login page or failed transiently,
   Phase 2 could miss products that CMR could still locate.

5. Misleading Phase 2 success criteria

   Phase 2 could previously continue even with zero useful OCO-2 downloads,
   leading to a later and less informative Phase 3 failure.

## Fixes Applied

- Disabled stale token usage in CURC shell scripts:
  - `unset GESDISC_DATA_TOKEN`
  - `unset EARTHDATA_USERNAME`
  - `unset EARTHDATA_PASSWORD`

- Added plain GES DISC URL fallback when tokenized GES DISC URLs fail.

- Narrowed Earthdata login-page classification so normal listing pages are not
  rejected just because they mention URS or OAuth.

- Updated CMR L1B lookup to prefer:
  - `short_name=OCO2_L1B_Science`
  - `version=11r` or `11.2r`
  - `provider=GES_DISC`

- Added CMR fallback for secondary OCO-2 products:
  - `OCO2_L2_Lite_FP`
  - `OCO2_L2_Met`
  - `OCO2_L2_CO2Prior`

- Added retry handling for transient HTTP statuses such as `429` and `5xx`.

- Tightened Phase 2 success criteria so a run with no usable OCO-2 files fails
  in Phase 2 with a precise error instead of surfacing later as a mysterious
  Phase 3 no-data problem.

## Expected Warnings

Some warnings are acceptable if they are followed by successful CMR fallback and
downloads:

- `GES DISC returned an Earthdata login page`
- `No Earthdata credentials - OCO-2 downloads may fail`

These warnings are mainly signals that the primary directory-listing path was
not usable. They are not fatal when followed by messages like:

- `Querying CMR for OCO-2 ...`
- `CMR found ... file ...`
- `Downloaded ...`
- `File exists locally ...`

Warnings are concerning when followed by:

- `No OCO-2 files downloaded`
- `No data found`
- nonzero `failed_download_count`
- Phase 2 failure

## Operational Script Notes

`curc_shell_blanca_general.sh` was returned from smoke-test mode to operational
mode after the download fixes were validated. The current operational flow runs:

```bash
python workspace/demo_combined.py --date "$date" --delete-modis --force-recompute
python src/spectral/fitting.py --date "$date"
```

The script keeps the auth cleanup and thread/HDF5 safeguards because those are
operationally relevant, not just test-only settings.

## Verification Already Performed

- Smoke test on `2016-09-15` with orbit `11734a` confirmed:
  - L1B located.
  - L2 Lite located through CMR fallback.
  - L2 Met located through CMR fallback.
  - L2 CO2Prior located through CMR fallback.
  - MODIS MYD35_L2 and MYD03 candidates found.

- Shell syntax check passed:

```bash
bash -n curc_shell_blanca_general.sh
```

