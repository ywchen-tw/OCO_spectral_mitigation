---
name: deplus-hetero-ablation
description: DE++ heterogeneous vs homogeneous M=10 ensemble ablation — no difference at scale; use homogeneous
metadata:
  type: project
---

DE++ experiment (`curc_shell_blanca_de_plus.sh`): does a HETEROGENEOUS M=10 deep ensemble (6 varied width/depth member specs) beat a HOMOGENEOUS M=10 (identical 64,32 members)? Motivated by a local 12-date single-fold ocean result showing hetero +0.013 tail-5% R² over homog at M=10.

**Verdict (2026-07-03): the two structures perform the same at full ~5M-row scale. Use HOMOGENEOUS M=10.**

Matched 5-fold ablation on the non-profile Jun-29 runs (`deplus_{ocean,land}_{hetero,homog}_f*`):
- Global bulk R²: ocean 0.590 (het) vs 0.586 (hom); land 0.593 vs 0.595 — indistinguishable.
- Near-cloud tail (the regime DE++ targeted): ocean hetero +0.004 to +0.006 R² across all tail groups but ~30× smaller than the fold std (±0.17) = noise; land hetero fractionally WORSE (−0.0003 to −0.0006).
- The motivating +0.013 local signal shrank to +0.005 (ocean) / ~0 (land) — **did not replicate**. Same "single-fold overfits, flattens at scale" pattern as [[kappa-feature-experiment]] and [[tabm-hpo-datekfold-lesson]].

**Why:** heterogeneity buys no measurable accuracy but adds the `--member_archs` config surface + per-member bookkeeping.

**How to apply:** keep the production/DE++ ensemble homogeneous M=10; don't run the `hetero_prof` arm (still commented out in the shell script at lines 110-111). Profile block is orthogonal — no reason to expect hetero_prof to differ. Reports: `results/model_comparison/deplus_hetero_ablation_{ocean,land}_noprof.md`.

Note: aggregate_folds only reads the mondrian global metrics JSON; the tail verdict REQUIRES reading the per-fold `de_mondrian_*_stratified_metrics.csv` (regimes: left_tail, near_cloud_tail).
