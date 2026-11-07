"""tccon_build_cases.py — build the per-case TCCON plot_data for one shard in ONE process.

Drop-in replacement for the serial bash `run_case` loop of the TCCON validation jobs
(fm1620_tccon.sh): same cases, same builder, same outputs, but
  * cases are sharded by DATE (--shard i --n-shards N), so a Slurm array runs them in parallel;
  * the builder module is imported once and its `_load_fold` is memoised, so the fold models
    are loaded once per shard instead of once per case;
  * cases sharing (date, surface) are built once; the other sites get a copy of the parquet
    (the builders' output depends only on input parquet + surface + models, not on the site);
  * outputs are written to <plot_data>.tmp then renamed, so a preempted case is never reused
    half-written.
Reuse rule unchanged: an existing plot_data newer than its input parquet is kept.

The builders are called through their own main() (argv per case), so results are identical
to the bash loop.  Per-case stdout/stderr still go to <case dir>/build.log.

Usage (inside the array task):
  python workspace/tccon_build_cases.py --kind xgb --out-base $OUT_BASE --csv-dir $RD/csv_collection \
      --ocean-model-dir ...f0 ...f4 --land-model-dir ...f0 ...f4 [--correction-base raw] \
      --shard $SLURM_ARRAY_TASK_ID --n-shards 8
  python workspace/tccon_build_cases.py ... --list     # print the shard's cases and exit
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import shlex
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

BUILDER = {'linreg': 'build_baseline_plot_data', 'xgb': 'build_baseline_plot_data',
           'tabm': 'build_tabm_plot_data', 'de': 'build_deepens_plot_data'}


def read_cases(script):
    """run_case lines of the case script, parsed as the bash loop does
    (text after the first '#' dropped; positional 1 date, 2 tccon, 9 surf, 11 site, 12 avail)."""
    cases = []
    for line in Path(script).read_text().splitlines():
        s = line.split('#', 1)[0].strip()
        if not s.startswith('run_case') or (len(s) > 8 and not s[8].isspace()):
            continue
        a = shlex.split(s)[1:]
        get = lambda k, d='': a[k - 1] if len(a) >= k and a[k - 1] != '' else d
        cases.append(dict(date=get(1), tccon=get(2), surf=get(9, 'both'),
                          site=get(11), avail=get(12, 'yes')))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=sorted(BUILDER), required=True)
    ap.add_argument('--script', default='curc_shell_blanca_plot_corr_xco2_deepens.sh')
    ap.add_argument('--out-base', required=True)
    ap.add_argument('--csv-dir', required=True)
    ap.add_argument('--ocean-model-dir', nargs='+', default=[])
    ap.add_argument('--land-model-dir', nargs='+', default=[])
    ap.add_argument('--correction-base', choices=('bc', 'raw'), default='bc')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    cases = [c for c in read_cases(args.script) if c['avail'] == 'yes']
    dates = sorted({c['date'] for c in cases})
    mine = {d for i, d in enumerate(dates) if i % args.n_shards == args.shard}
    cases = [c for c in cases if c['date'] in mine]
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} date(s), {len(cases)} case(s)", flush=True)
    if args.list:
        for c in cases:
            print(f"  {c['date']} {c['site'] or '-'} {c['surf']}")
        return

    mod = importlib.import_module(BUILDER[args.kind])
    mod._load_fold = functools.lru_cache(maxsize=None)(mod._load_fold)   # load each fold once

    out_base, csv_dir = Path(args.out_base), Path(args.csv_dir)
    built = {}          # (date, surf) -> plot_data path built in this shard
    n_ok = n_reuse = n_copy = n_fail = n_skip = 0
    for c in cases:
        date, surf, site = c['date'], c['surf'], c['site']
        inp = csv_dir / f"combined_{date}_all_orbits.parquet"
        outdir = out_base / (f"combined_{date}_{site}" if site else f"combined_{date}_all_orbits")
        pdat = outdir / 'plot_data.parquet'
        tag = f"CASE {date} {site} ({surf})"
        if not inp.exists():
            print(f"{tag}: SKIP no parquet", flush=True); n_skip += 1; continue
        if pdat.exists() and pdat.stat().st_mtime > inp.stat().st_mtime:
            print(f"{tag}: reuse", flush=True); n_reuse += 1
            built.setdefault((date, surf), pdat); continue
        outdir.mkdir(parents=True, exist_ok=True)
        tmp = pdat.with_name(pdat.name + '.tmp')
        src = built.get((date, surf))
        if src is not None and src.exists():
            shutil.copyfile(src, tmp); tmp.replace(pdat)
            shutil.copyfile(src.parent / 'build.log', outdir / 'build.log') \
                if (src.parent / 'build.log').exists() else None
            print(f"{tag}: copy of {src.parent.name}", flush=True); n_copy += 1; continue

        argv = [BUILDER[args.kind] + '.py']
        if surf in ('both', 'ocean') and args.ocean_model_dir:
            argv += ['--ocean-model-dir', *args.ocean_model_dir]
        if surf in ('both', 'land') and args.land_model_dir:
            argv += ['--land-model-dir', *args.land_model_dir]
        if args.kind in ('linreg', 'xgb'):
            argv += ['--model-kind', args.kind, '--correction-base', args.correction_base]
        if args.kind == 'de':
            argv += ['--emit-members']
        argv += ['--input', str(inp), '--output', str(tmp)]
        t0 = time.time()
        ok = False
        with open(outdir / 'build.log', 'w') as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            sys.argv = argv
            try:
                mod.main(); ok = True
            except SystemExit as e:
                ok = e.code in (None, 0)
                print(f"SystemExit: {e.code}")
            except Exception:
                traceback.print_exc()
        if ok and tmp.exists():
            tmp.replace(pdat); built[(date, surf)] = pdat; n_ok += 1
            print(f"{tag}: built in {time.time() - t0:.1f} s", flush=True)
        else:
            tmp.unlink(missing_ok=True); n_fail += 1
            print(f"{tag}: build failed (see {outdir}/build.log)", flush=True)
            print('  ' + ''.join((outdir / 'build.log').read_text().splitlines(True)[-3:]), flush=True)
    print(f"shard {args.shard} done: built {n_ok}, copied {n_copy}, reused {n_reuse}, "
          f"skipped {n_skip}, failed {n_fail}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
