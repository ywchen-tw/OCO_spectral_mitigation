#!/usr/bin/env python3
"""TabM random-search HPO — CURC-ready, date-blocked validation.

Unlike the local single-date/random-split proxy (which overfit: its winner LOST to
the default on date_kfold), this evaluates every trial with --val_split date_kfold on
the FULL dataset, so the search objective IS held-out-date generalization — the thing
we deploy on.

Parallelism: run as a SLURM array; each task passes a distinct --seed so it samples a
DIFFERENT slice of the space and writes its own seed-tagged CSV.  Aggregate afterwards
with --aggregate.

Trial mode:
  python tabm_hpo_search.py --seed 0 --n_trials 5 --sfc_type 0 --feature_set full_contam \
      --val_split date_kfold --n_folds 5 --fold 0 --epochs 60
Aggregate mode (after all array tasks finish):
  python tabm_hpo_search.py --aggregate --tag hpo_dk_ocean_full_contam
"""
import argparse, json, os, subprocess, sys, random, math, time, csv, glob, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABM_BASE = ROOT / "results/model_tabm"
COMP_DIR  = ROOT / "results/model_comparison"

# ── Search space — PRODUCTION scale.  Batches are large (small batches are infeasible
# on the ~10M-row set and were what the local proxy over-rewarded).  weight_decay and
# dropout ranges are wide so date-blocked CV can pick MORE regularization if it
# generalizes better (the random-split winner used wd=8e-6, near zero). ───────────────
def sample(rng):
    return {
        "K":            rng.choice([8, 16, 32]),
        "d_model":      rng.choice([128, 192, 256, 384]),
        "n_layers":     rng.choice([2, 3, 4]),
        "dropout":      round(rng.uniform(0.0, 0.40), 3),
        "lr":           round(10 ** rng.uniform(math.log10(3e-4), math.log10(5e-3)), 6),
        "weight_decay": round(10 ** rng.uniform(math.log10(1e-6), math.log10(3e-3)), 7),
        "batch_size":   rng.choice([4096, 8192, 16384]),
        "huber_delta":  round(rng.uniform(0.5, 2.0), 2),
    }

def build_config(hp, epochs, patience):
    # Set BOTH platform blocks so the same config runs on CURC (linux) or locally.
    return {
        "model": {"K": hp["K"], "d_model": hp["d_model"],
                  "n_layers": hp["n_layers"], "dropout": hp["dropout"]},
        "train": {"darwin_epochs": epochs, "linux_epochs": epochs,
                  "darwin_batch_size": hp["batch_size"], "linux_batch_size": hp["batch_size"],
                  "lr": hp["lr"], "weight_decay": hp["weight_decay"],
                  "patience": patience, "log_every": 25, "seed": 42},
        "loss":  {"loss": "huber", "huber_delta": hp["huber_delta"]},
    }

def read_metric(out_dir):
    cands = [p for p in Path(out_dir).glob("*_metrics.json")
             if "stratified" not in p.name and "rearranged" not in p.name]
    if not cands:
        return None
    g = json.load(open(sorted(cands)[0])).get("global", {})
    return {"r2": g.get("r2"), "rmse": g.get("rmse"), "mae": g.get("mae"),
            "cov90": g.get("coverage_90")}

# ── Aggregate every seed-tagged CSV for a tag, rank by held-out R² ─────────────────────
def aggregate(tag):
    rows = []
    for c in sorted(COMP_DIR.glob(f"{tag}_s*_trials.csv")):
        rows += list(csv.DictReader(open(c)))
    if not rows:
        print(f"No trial CSVs found for tag '{tag}' in {COMP_DIR}"); return
    rows.sort(key=lambda r: float(r["r2"]), reverse=True)
    out = COMP_DIR / f"{tag}_AGG.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n=== {tag}: {len(rows)} trials (date_kfold) ===")
    print(f"{'rank':>4} {'R²':>7} {'RMSE':>7} {'K':>3} {'d_model':>7} {'L':>2} {'drop':>5} "
          f"{'lr':>8} {'wd':>9} {'bs':>6} {'hub':>4}")
    for i, d in enumerate(rows[:12]):
        print(f"{i+1:>4} {float(d['r2']):>7.4f} {float(d['rmse']):>7.4f} {d['K']:>3} {d['d_model']:>7} "
              f"{d['n_layers']:>2} {float(d['dropout']):>5.2f} {float(d['lr']):>8.5f} "
              f"{float(d['weight_decay']):>9.6f} {d['batch_size']:>6} {float(d['huber_delta']):>4.1f}")
    best = rows[0]
    # Write the tuned config straight to where the comparison script expects it.
    cfg = build_config({k: (int(best[k]) if k in ("K","d_model","n_layers","batch_size")
                            else float(best[k]))
                        for k in ("K","d_model","n_layers","dropout","lr","weight_decay",
                                  "batch_size","huber_delta")},
                       epochs=int(best.get("epochs", 60)), patience=15)
    tuned_path = ROOT / "tabm_tuned_ocean_datekfold.json"
    json.dump(cfg, open(tuned_path, "w"), indent=2)
    print(f"\nBEST date_kfold R²={float(best['r2']):.4f}  -> wrote {tuned_path.name}")
    print(f"Full ranking: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="search RNG seed (vary per array task)")
    ap.add_argument("--sfc_type", type=int, default=0)
    ap.add_argument("--feature_set", type=str, default="full_contam")
    ap.add_argument("--val_split", type=str, default="date_kfold")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--data", type=str, default=None, help="override input parquet")
    args = ap.parse_args()

    sfc_name = "ocean" if args.sfc_type == 0 else "land"
    tag = args.tag or f"hpo_dk_{sfc_name}_{args.feature_set}"
    if args.aggregate:
        aggregate(tag); return

    rng = random.Random(args.seed)
    cfg_dir = TABM_BASE / "_hpo_dk_configs"; cfg_dir.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = COMP_DIR / f"{tag}_s{args.seed}_trials.csv"

    trials = []
    for t in range(args.n_trials):
        hp = sample(rng)
        suffix = f"{tag}_s{args.seed}_t{t:02d}"
        cfg_path = cfg_dir / f"{suffix}.json"
        json.dump(build_config(hp, args.epochs, args.patience), open(cfg_path, "w"), indent=2)
        cmd = [sys.executable, "-m", "models.tabm", "--sfc_type", str(args.sfc_type),
               "--suffix", suffix, "--feature_set", args.feature_set,
               "--val_split", args.val_split, "--n_folds", str(args.n_folds),
               "--fold", str(args.fold), "--config", str(cfg_path)]
        if args.data:
            cmd += ["--data", args.data]
        env = {**os.environ, "PYTHONPATH": f"src{os.pathsep}{os.environ.get('PYTHONPATH','')}"}
        t0 = time.monotonic()
        print(f"\n[s{args.seed} {t+1}/{args.n_trials}] {suffix}  {hp}", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        m = read_metric(TABM_BASE / suffix)
        if m is None:
            print(f"  FAILED ({time.monotonic()-t0:.0f}s) rc={r.returncode}", flush=True); continue
        trials.append({"trial": t, "suffix": suffix, **hp, **m,
                       "epochs": args.epochs, "sec": round(time.monotonic()-t0)})
        print(f"  date_kfold R²={m['r2']:.4f} RMSE={m['rmse']:.4f} "
              f"({time.monotonic()-t0:.0f}s)", flush=True)

    if trials:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trials[0].keys())); w.writeheader(); w.writerows(trials)
        print(f"\nWrote {log_path}  ({len(trials)} trials)")

if __name__ == "__main__":
    main()
