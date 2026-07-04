# TCCON Comparison Bibliography + ML Model Improvement Ideas

**Date:** 2026-07-03
**Companion to:** `log/PROJECT_REVIEW_2026-07-03.md`

---

## Part 1 — TCCON comparison literature

All DOIs verified against Crossref / journal landing pages (2026-07-03). 45 entries, 7 clusters.

### Cluster 1 — TCCON network + data protocol

- **Wunch, D., et al. (2011).** The Total Carbon Column Observing Network. *Phil. Trans. R. Soc. A*, 369, 2087–2112. DOI: 10.1098/rsta.2010.0240 — Canonical TCCON overview; primary citation for TCCON as validation truth standard. **MUST-CITE**
- **Laughner, J. L., et al. (2024).** The Total Carbon Column Observing Network's GGG2020 data version. *ESSD*, 16, 2197–2260. DOI: 10.5194/essd-16-2197-2024 — The GGG2020 product the collocation pipeline consumes; mandatory data citation. **MUST-CITE**
- **Rodgers, C. D., & Connor, B. J. (2003).** Intercomparison of remote sounding instruments. *JGR-Atmos*, 108(D3), 4116. DOI: 10.1029/2002JD002299 — The averaging-kernel/prior harmonization formalism any satellite-vs-TCCON comparison must follow. **MUST-CITE**
- **Wunch, D., et al. (2010).** Calibration of TCCON using aircraft profile data. *AMT*, 3, 1351–1362. DOI: 10.5194/amt-3-1351-2010 — Ties TCCON XCO2 to the WMO in-situ scale (absolute-accuracy claim of the reference).
- **Messerschmidt, J., et al. (2011).** Calibration of TCCON column-averaged CO2: first aircraft campaign over European TCCON sites. *ACP*, 11, 10765–10777. DOI: 10.5194/acp-11-10765-2011 — Independent aircraft calibration establishing station-to-station consistency.

### Cluster 2 — OCO-2/ACOS vs TCCON validation + operational bias correction

- **Wunch, D., et al. (2017).** Comparisons of OCO-2 XCO2 measurements with TCCON. *AMT*, 10, 2209–2238. DOI: 10.5194/amt-10-2209-2017 — The canonical OCO-2-vs-TCCON validation methodology (coincidence criteria, bias metrics) our TCCON pipeline builds on. **MUST-CITE**
- **O'Dell, C. W., et al. (2018).** Improved retrievals of CO2 from OCO-2 with the version 8 ACOS algorithm. *AMT*, 11, 6539–6576. DOI: 10.5194/amt-11-6539-2018 — Defines ACOS + the operational parametric bias correction the ML correction is positioned against. **MUST-CITE**
- **Kiel, M., et al. (2019).** How bias correction goes wrong: XCO2 affected by erroneous surface pressure estimates. *AMT*, 12, 2241–2259. DOI: 10.5194/amt-12-2241-2019 — State-correlated error corrupting the operational bias correction; same failure mode motivating a cloud-proximity-aware correction.
- **Kulawik, S., et al. (2016).** Consistent evaluation of ACOS-GOSAT, BESD, CarbonTracker, and MACC through comparisons to TCCON. *AMT*, 9, 683–709. DOI: 10.5194/amt-9-683-2016 — TCCON-anchored systematic-vs-random error framework.
- **Crisp, D., et al. (2017).** On-orbit performance of the OCO-2 instrument. *AMT*, 10, 59–81. DOI: 10.5194/amt-10-59-2017 — L1B glint-mode radiance/calibration reference.
- **Eldering, A., et al. (2017).** OCO-2: first 18 months of science data products. *AMT*, 10, 549–563. DOI: 10.5194/amt-10-549-2017 — Primary mission/data-product citation.
- **Das, S., et al. (2025).** Comparisons of v11.1 OCO-2 XCO2 with GGG2020 TCCON. *Earth and Space Science*, 12, e2024EA003935. DOI: 10.1029/2024EA003935 — Current-generation (v11.1 vs GGG2020) validation baseline our corrected product must be benchmarked against.

### Cluster 3 — Collocation / coincidence-criteria methodology

- **Wunch, D., et al. (2011).** A method for evaluating bias in global measurements of CO2 total columns from space. *ACP*, 11, 12317–12337. DOI: 10.5194/acp-11-12317-2011 — Dynamic (potential-temperature) coincidence criterion; standard alternative to geometric collocation. **MUST-CITE** (note: ACP, not GRL)
- **Nguyen, H., et al. (2014).** A method for colocating satellite XCO2 to ground-based data (ACOS-GOSAT/TCCON). *AMT*, 7, 2631–2644. DOI: 10.5194/amt-7-2631-2014 — Geostatistical collocation; shows criterion choice materially changes validation stats (justifies 100/50 km radius sensitivity).
- **Hedelius, J. K., et al. (2017).** Intercomparability of XCO2 and XCH4 from the US TCCON sites. *AMT*, 10, 1481–1493. DOI: 10.5194/amt-10-1481-2017 — Station-scale representation/mismatch error = the noise floor of satellite–TCCON comparisons.
- **Keppel-Aleks, G., et al. (2011).** Sources of variations in total column carbon dioxide. *ACP*, 11, 3581–3593. DOI: 10.5194/acp-11-3581-2011 — Synoptic-scale gradients dominate column variability; physical basis of coincidence-criterion design.
- **Inoue, M., et al. (2016).** Bias corrections of GOSAT SWIR XCO2/XCH4 with TCCON and evaluation using aircraft data. *AMT*, 9, 3491–3512. DOI: 10.5194/amt-9-3491-2016 — Precedent for correction-then-independent-validation design.

### Cluster 4 — High-latitude / difficult-regime validation

- **Jacobs, N., et al. (2020).** Quality controls, bias, and seasonality of CO2 columns in the boreal forest (OCO-2, TCCON, EM27/SUN). *AMT*, 13, 5033–5063. DOI: 10.5194/amt-13-5033-2020 — Benchmark high-lat validation; documents QC throughput loss in exactly the regime where the correction claims skill.
- **Mendonca, J., et al. (2021).** Feasibility of a neural network to filter OCO-2 retrievals at northern high latitudes. *AMT*, 14, 7511–7524. DOI: 10.5194/amt-14-7511-2021 — Closest high-lat precedent: NN quality *filtering* vs our *correction*.
- **Jacobs, N., et al. (2021).** Spatial distributions of XCO2 seasonal cycle amplitude/phase over northern high latitudes. *ACP*, 21, 16661–16687. DOI: 10.5194/acp-21-16661-2021 — Residual seasonal biases; payoff argument for recovering near-cloud/snow-season soundings.
- **Kivi, R., & Heikkinen, P. (2016).** FTS measurements of column CO2 at Sodankylä. *GI*, 5, 271–279. DOI: 10.5194/gi-5-271-2016 — Sodankylä (67.4°N) station reference.
- **Batchelor, R. L., et al. (2009).** A new Bruker IFS 125HR FTIR for PEARL, Eureka. *JTECH*, 26, 1328–1340. DOI: 10.1175/2009JTECHA1215.1 — Eureka (80°N) instrument reference.
- **Buschmann, M., et al. (2016).** Retrieval of xCO2 from ground-based mid-IR (NDACC) spectra and comparison to TCCON. *AMT*, 9, 577–585. DOI: 10.5194/amt-9-577-2016 — Ny-Ålesund (79°N) xCO2 capability/cross-network consistency.

### Cluster 5 — 3D cloud effects on XCO2 (closest prior work)

- **Merrelli, A., et al. (2015).** Estimating bias in OCO-2 caused by 3-D radiation scattering from unresolved boundary layer clouds. *AMT*, 8, 1641–1656. DOI: 10.5194/amt-8-1641-2015 — Foundational simulation of the 3D-scattering mechanism (up to ~5 ppm).
- **Massie, S. T., et al. (2017).** Observational evidence of 3-D cloud effects in OCO-2 CO2 retrievals. *JGR-Atmos*, 122, 7064–7085. DOI: 10.1002/2016JD026111 — First observational demonstration that XCO2 error depends on cloud proximity. (JGR, not AMT.)
- **Massie, S. T., et al. (2021).** Analysis of 3D cloud effects in OCO-2 XCO2 retrievals. *AMT*, 14, 1475–1499. DOI: 10.5194/amt-14-1475-2021 — Introduced the MODIS nearest-cloud-distance analysis (~40% of QF0 soundings within 4 km of cloud); the methodological template this pipeline extends. **MUST-CITE**
- **Massie, S. T., et al. (2023).** Insights into 3D cloud radiative transfer effects for OCO. *AMT*, 16, 2145–2166. DOI: 10.5194/amt-16-2145-2023 — Residual near-cloud biases (0–2.5 ppm) vs cloud distance in the operational product; defines the residual our model corrects.
- **Mauceri, S., Massie, S., & Schmidt, S. (2023).** Correcting 3D cloud effects in XCO2 retrievals from OCO-2. *AMT*, 16, 1461–1476. DOI: 10.5194/amt-16-1461-2023 — THE closest prior ML cloud-bias correction (20% land / 40% ocean variability reduction); explicit comparison required. **MUST-CITE**
- **Chen, Y.-W., et al. (2025).** Mitigation of OCO-2 CO2 biases in the vicinity of clouds with 3D calculations using EaR3T. *AMT*, 18, 1859–1884. DOI: 10.5194/amt-18-1859-2025 — Our own physics-based (3D RT) near-cloud mitigation; position the ML approach against it.
- **Emde, C., et al. (2022).** Impact of 3D cloud structures on atmospheric trace gas products from UV–Vis sounders — Part 1. *AMT*, 15, 1587–1608. DOI: 10.5194/amt-15-1587-2022 — Generality of the cloud-adjacency mechanism beyond OCO-2.

### Cluster 6 — ML bias correction / ML XCO2 retrieval validated with TCCON

- **David, L., Bréon, F.-M., & Chevallier, F. (2021).** XCO2 estimates from OCO-2 using a neural network approach. *AMT*, 14, 117–132. DOI: 10.5194/amt-14-117-2021 — First end-to-end NN XCO2 retrieval for OCO-2 with TCCON validation.
- **Bréon, F.-M., et al. (2022).** On the potential of a neural-network-based approach for estimating XCO2 from OCO-2. *AMT*, 15, 5219–5234. DOI: 10.5194/amt-15-5219-2022 — NN pitfalls (location/time proxies, mimicking training data); motivates our feature design and validation guards.
- **Keely, W. R., et al. (2023).** A nonlinear data-driven approach to bias correction of XCO2 for ACOS v10. *AMT*, 16, 5725–5748. DOI: 10.5194/amt-16-5725-2023 — Nonlinear ML bias correction beating the operational linear one (+14% throughput); direct methodological sibling. **MUST-CITE**
- **Noël, S., et al. (2021).** XCO2 retrieval for GOSAT/GOSAT-2 based on FOCAL. *AMT*, 14, 3837–3869. DOI: 10.5194/amt-14-3837-2021 — Random-forest quality filtering, TCCON-validated (cross-mission context).
- **Noël, S., et al. (2022).** Retrieval of greenhouse gases from GOSAT/GOSAT-2 (FOCAL v3). *AMT*, 15, 3401–3437. DOI: 10.5194/amt-15-3401-2022 — Updated multi-gas product (cite one or both per space).
- **Xie, F., et al. (2024).** Fast retrieval of XCO2 over east Asia based on OCO-2 spectral measurements. *AMT*, 17, 3949–3967. DOI: 10.5194/amt-17-3949-2024 — Recent MLP-based, TCCON-validated retrieval.
- **Mauceri, S., et al. (2025).** Uncertainty-Aware ML Bias Correction and Filtering for OCO-2: 1. *Earth and Space Science*, 12, e2025EA004328. DOI: 10.1029/2025EA004328 — Latest TCCON-constrained ML bias correction; cite + differentiate. **SCOOP-RISK: HIGH**
- **Keely, W., et al. (2025).** Uncertainty-Aware ML Bias Correction and Filtering for OCO-2: 2. *Earth and Space Science*, 12, e2025EA004329. DOI: 10.1029/2025EA004329 — Companion uncertainty-aware ML filtering; cite + differentiate. **SCOOP-RISK: HIGH**

### Cluster 7 — OCO-3 / other sensors vs TCCON + EM27/COCCON

- **Taylor, T. E., et al. (2020).** OCO-3 early mission operations and initial (vEarly) XCO2 and SIF retrievals. *RSE*, 251, 112032. DOI: 10.1016/j.rse.2020.112032 — Supports transferability of the correction to OCO-3.
- **Taylor, T. E., et al. (2023).** Consistency between OCO-2 and OCO-3 XCO2 (ACOS v10). *AMT*, 16, 3173–3209. DOI: 10.5194/amt-16-3173-2023 — Cross-sensor consistency vs TCCON (AMT, not ESSD).
- **Frey, M., et al. (2019).** Building the COllaborative Carbon Column Observing Network (COCCON). *AMT*, 12, 1513–1530. DOI: 10.5194/amt-12-1513-2019 — Legitimizes EM27/SUN as supplementary validation.
- **Sha, M. K., et al. (2020).** Intercomparison of low- and high-resolution IR spectrometers for CO2/CH4/CO columns. *AMT*, 13, 4791–4839. DOI: 10.5194/amt-13-4791-2020 — EM27-vs-TCCON accuracy chain.
- **Hedelius, J. K., et al. (2016).** Errors and biases in retrievals from a 0.5 cm⁻¹ resolution solar-viewing spectrometer. *AMT*, 9, 3527–3546. DOI: 10.5194/amt-9-3527-2016 — EM27/SUN error budget.
- **Yoshida, Y., et al. (2013).** Improved GOSAT SWIR XCO2/XCH4 retrieval and TCCON validation. *AMT*, 6, 1533–1547. DOI: 10.5194/amt-6-1533-2013 — Canonical GOSAT-vs-TCCON precedent.
- **Suto, H., et al. (2021).** TANSO-FTS-2 on GOSAT-2 during its first year in orbit. *AMT*, 14, 2013–2039. DOI: 10.5194/amt-14-2013-2021 — GOSAT-2 instrument reference.

### MUST-CITE shortlist (reviewer expectations)

1. Wunch et al. 2011 (PTRSA) — TCCON network
2. Laughner et al. 2024 (ESSD) — GGG2020
3. Rodgers & Connor 2003 (JGR) — AK formalism
4. Wunch et al. 2017 (AMT) — OCO-2 vs TCCON protocol
5. O'Dell et al. 2018 (AMT) — ACOS + operational bias correction
6. Massie et al. 2021 (AMT) — nearest-cloud-distance methodology (with Massie 2017 as origin)
7. Mauceri et al. 2023 (AMT) — closest prior ML 3D-cloud correction
8. Keely et al. 2023 (AMT) — closest prior general ML bias correction

### 2024–2026 scoop-risk / overlap flags

- **HIGH — Mauceri et al. 2025 + Keely et al. 2025** (ESS Parts 1+2, same Mauceri/Keely/O'Dell group): uncertainty-aware, TCCON-constrained ML bias correction + filtering for OCO-2. They target *general* bias correction, not cloud-proximity-specific — our framing (path-length cumulants + cloud-distance) remains distinct, but both must be cited and differentiated in the introduction.
- **MODERATE** — EGUsphere egusphere-2024-3990 (GOSAT-2 RemoTeC + ML quality filtering, in discussion); cite only if accepted.
- **LOW** — arXiv:2504.17074 (diffusion-based XCO2 retrieval); 2026 arXiv ML-emulation preprint; adjacent, monitor.
- **Watch** — Das et al. "OCO-2 v11.2/OCO-3 v11 vs COCCON" (submitted, no DOI); re-check at revision time.
- No 2024–2026 paper found doing ML correction of *cloud-proximity* XCO2 bias other than the Mauceri/Keely line and Chen et al. 2025 (our own).

### Dropped as unverifiable / not peer-reviewed

Payne et al. "B11" (ATBD only); Osterman et al. Lite-file docs (product documentation); Kulawik et al. 2019 AMTD (withdrawn); a Buschmann Ny-Ålesund-vs-satellite paper (does not exist); Wunch et al. 2015 GGG2014 report (CaltechDATA tech report — cite as dataset if needed).

---

## Part 2 — ML model improvements for the anomaly regression problem

**Problem shape:** per-sounding regression, target = within-orbit clear-sky XCO2 anomaly (ppm); ~30–50 tabular features per surface; heavy imbalance (~81% far-cloud rows, skill matters most in the near-cloud tail); heteroscedastic noise; honest date_kfold R² ≈ 0.53 ocean / 0.39 land. Raw upstream data are *arrays* (3×1016-channel spectra → ln T vs τ curves; 72-level profiles) currently compressed to hand-crafted scalars (k1/k2 cumulants, EOF PCs).

### 2.0 First, know the ceiling

Before trying new architectures, estimate the **label-noise ceiling**. The target inherits sampling error from the clear-sky reference mean (reference std ≤ 1.0 ppm over a finite neighbor count n → ref-mean error ~ std/√n, plus real sub-0.25° CO2 variability). Compute the implied irreducible variance per row (you have `ref_std`, `ref_count` in the parquet) and translate to a max attainable R². Evidence the ceiling is near: HPO landscape flat, TabM ≈ MLP under date CV, M=10 and heterogeneous ensembles no better than M=5 homogeneous. **Architecture is probably not the bottleneck; label noise and distribution shift are.** New models should target the *tail* and *new information*, not global R².

### 2.1 Along-track spatial context (sequence model) — **DEMOTED: violates deployment constraint**

**Project constraint (2026-07-04):** the correction must run footprint-by-footprint at inference — no cloud info AND no neighboring-footprint info as inputs. That per-footprint independence is itself a key selling point (deployable everywhere, no MODIS, no swath context). Sequence/context models are therefore NOT production candidates; keep them only as a diagnostic upper bound ("how much does spatial context add?") to quantify what the per-footprint model leaves on the table.

OCO-2 soundings form an 8 (footprint) × N (along-track) grid per orbit; the anomaly and its causes (cloud fields) are spatially coherent. Feed a window of consecutive soundings (e.g., 8×64 grid of the per-sounding feature vectors, or just the spectral/cumulant channels) into a small 2D CNN / 1D transformer with the center sounding as the regression target.

- **Why it matters scientifically:** the model can learn cloud proximity *from OCO-2 data alone* — no MODIS needed. This directly addresses (a) the post-2022 Aqua free-drift `NoMODIS` gap, and (b) deployability of the correction where collocation is unavailable. "MODIS-free cloud-proximity correction" is a headline capability, not an incremental R² gain.
- **Leakage caution (critical):** neighboring soundings within ±0.25° lat may be members of the *label's clear-sky reference set*. Exclude reference-set soundings from the input window, or mask the `xco2`-bearing channels of neighbors (feed only spectral/geometry channels of the context). Validate with orbit-blocked CV.
- Cheap variant first: add hand-rolled neighborhood aggregates (rolling mean/std/gradient of k1, k2, dp_abp, continuum over ±10 along-track soundings) as tabular features to the existing DE. If those already help, the sequence model will help more.

### 2.2 Array-native encoders: learn from the ln T curves directly

The (7,3,7) cumulant fit is a lossy, hand-crafted compression. Test what it discards:

- **Hybrid residual encoder (recommended design):** keep k1/k2 features, and *additionally* encode the fit residual curve `ln T − polynomial(τ)` per band with a small 1D CNN (input: residual interpolated onto a fixed τ grid, or (τᵢ, residualᵢ) pairs through a set/point-transformer to handle T>1-masked channels without interpolation). Concatenate the learned embedding (8–16 dims/band) with tabular features → existing DE head.
  - If the embedding adds nothing → strong paper claim: "the cumulant parameters capture the cloud-relevant spectral information."
  - If it helps in the near-cloud tail → new signal beyond the gamma-path-PDF model (e.g., non-gamma path distributions near cloud edges).
- Alternative inputs: raw per-band spectra sorted by τ; L2 EOF-residual amplitudes (already partly in features as `eof3_1_rel`).
- **Risks:** learned encoders can latch onto instrument/footprint/date fingerprints → orbit- and date-blocked CV mandatory; interpretability cost — keep the hybrid framing so the physics story survives.

### 2.3 Tail-focused objectives (cheap, try soon)

- **Mixture density network head** (2–3 Gaussian components) instead of single-Gaussian beta-NLL: near-cloud anomalies are skewed/heavy-tailed; a single symmetric likelihood pulls mu toward the far-cloud mode.
- **Multi-quantile / pinball head** (predict q05…q95 jointly) — pairs naturally with the existing conformal wrapper; XGBoost quantile as a cheap baseline.
- **Two-stage / expert gating:** train global model → fine-tune a near-cloud expert (cld_dist < gate) and blend with the existing `p_near` soft gate from the XGB cloud classifier (mixture-of-experts, learned or fixed gate). Note prior result: full mu beat P·mu and hard gate — so blend at the *representation* level (shared trunk, expert heads), not the output level.

### 2.4 Use the unlabeled rows: self-supervised pretraining

The `std_thres` filter drops label-less soundings — disproportionately the hardest heavily-cloudy scenes (selection effect flagged in the review as M1). Those rows still have full feature vectors:

- Pretrain the trunk with masked-feature reconstruction or contrastive learning on *all* soundings (labeled + unlabeled), then fine-tune the regression head on labeled rows. This is the only route that lets the hardest scenes shape the representation.
- Lighter variant: pseudo-labeling / consistency regularization on unlabeled near-cloud rows.

### 2.5 Alternative tabular architectures (low expected gain — baselines only)

- **FT-Transformer / tabular ResNet** (Gorishniy et al. 2021): worth one run each as baselines; literature and this project's own evidence (TabM ≈ MLP) suggest parity, not wins.
- **TabPFN v2:** in-context prior-fitted network, strong on ≤10k-row problems. Not viable for the full dataset, but interesting for per-region/per-season few-shot fits and as a fast sanity baseline on the near-cloud subset.
- **GBDT with the new features** (neighborhood aggregates from 2.1, embeddings from 2.2): XGBoost remains the honesty check for any feature-level idea.

### 2.6 Uncertainty (already good — minor options)

Mondrian conformal on the DE is solid. Cheaper epistemic alternatives if compute matters: last-layer Laplace or SNGP on a single network. Skip deep evidential regression (known calibration pathologies). Keep coverage reporting per cloud-distance bin.

### 2.7 What NOT to spend time on (already tested, no win)

- More HPO on TabM/MLP (flat landscape; default TabM beat tuned ones).
- Bigger ensembles (M=10 tied M=5) and heterogeneous member architectures (tied homogeneous at scale).
- `snow_flag` as a feature (neutral); κ = k1²/k2 feature (slightly hurts).

### 2.8 Suggested experiment order

| # | Experiment | Cost | What success buys |
|---|---|---|---|
| 1 | Label-noise ceiling estimate | hours | Know when to stop chasing R² |
| 2 | Neighborhood aggregate features → DE/XGB | days | Cheap test of spatial-context value |
| 3 | MDN / multi-quantile head | days | Tail skill, better near-cloud likelihood |
| 4 | Along-track sequence model (8×N grid CNN/transformer) | 1–2 wks | MODIS-free correction (headline capability) |
| 5 | Hybrid residual-curve encoder | 1–2 wks | Tests cumulant-fit sufficiency; paper ablation |
| 6 | Self-supervised pretraining on unlabeled rows | 1–2 wks | Coverage of hardest scenes (answers review M1 selection effect) |
| 7 | FT-Transformer / TabPFN baselines | days | Reviewer-proof baseline table |

All of the above under **date_kfold + orbit-blocked CV**, evaluated globally *and* on near-cloud / bottom-5% tail slices, with the TCCON chain as the final arbiter (per project policy).
