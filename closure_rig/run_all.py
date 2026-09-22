"""Batch runner: compatibility audit plus experiments over every config.

    python -m closure_rig.run_all --out results/ [--fluoddity-path ../Fluoddity] [--quick]

Always writes results/config_audit.csv, even if the experiment sweep is skipped,
because the audit is what says which configs are being run under semantics they
weren't authored for.
"""

import argparse
import csv
import json
import os
import traceback

from . import configs as cfg_mod
from . import e0 as e0_mod
from . import run as run_mod
from .harness import Rig, create_context

# Candidate locations for the sibling repo. Its Core presets are already
# vendored into physics_configs/, so a missing checkout is a warning, not an error.
FLUODDITY_CANDIDATES = ['../Fluoddity', '../fluoddity', '../aphid91/fluoddity', '~/aphid91/fluoddity']


def find_fluoddity(explicit=None):
    for c in ([explicit] if explicit else []) + FLUODDITY_CANDIDATES:
        p = os.path.expanduser(c)
        if os.path.isdir(os.path.join(p, 'physics_configs', 'Core')):
            return p
    return None


def write_audit(entries, out):
    rows = [cfg_mod.audit_row(e) for e in entries]
    path = os.path.join(out, 'config_audit.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results')
    ap.add_argument('--fluoddity-path', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--configs', nargs='*', default=None,
                    help='config names to run; default is all discovered')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warmup-cap', type=int, default=30000)
    ap.add_argument('--probe-every', type=int, default=100)
    ap.add_argument('--warmup-window', type=int, default=20,
                    help='probes in the trailing split-half drift window')
    ap.add_argument('--skip-warmup', action='store_true')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    fpath = find_fluoddity(args.fluoddity_path)
    if fpath:
        print(f"Fluoddity checkout: {fpath}")
    else:
        print("Fluoddity checkout not found. Its Core presets are already vendored "
              "into physics_configs/, so config coverage is unaffected; only the "
              "upstream shaders (used to document ignored settings) are missing.\n"
              "  git clone --depth 1 https://github.com/aphid91/Fluoddity ../Fluoddity")

    entries = cfg_mod.discover('.', fluoddity_path=fpath)
    path, rows = write_audit(entries, args.out)
    approx = sum(r['core_semantics_approximation'] for r in rows)
    print(f"{len(entries)} unique configs -> {path}")
    print(f"  {approx} tagged core_semantics_approximation, "
          f"{sum(r['breaks_e3_isotropy'] for r in rows)} would break E3 isotropy upstream")
    if args.audit_only:
        return

    wanted = entries
    if args.configs:
        want = set(args.configs)
        wanted = [e for e in entries if e['name'] in want]
        missing = want - {e['name'] for e in wanted}
        if missing:
            print(f"  WARNING: not found: {sorted(missing)}")

    ctx = create_context()          # one context reused across configs

    # E0.2 depends on no config uniform (nothing from the config reaches
    # brush.frag, and the splat size is a #define), so measure it once.
    blend = None
    if wanted:
        print('\nE0.2 blend check (config-independent, run once) ...', flush=True)
        blend = e0_mod.blend_check(Rig(wanted[0]['path'], ctx=ctx))
        with open(os.path.join(args.out, 'e0_blend_check.json'), 'w') as f:
            json.dump(blend, f, indent=2)
        sp = blend['subpixel_sweep']
        print(f"  deposit varies {sp['max_over_min']:.0f}x with sub-pixel position "
              f"(CV {sp['cv_percent']:.0f}%); velocity still exact to "
              f"{sp['max_vel_deviation']:.1e}", flush=True)

    results = {}
    for e in wanted:
        print(f"\n=== {e['name']} ({e['source']}) ===", flush=True)
        try:
            rig = Rig(e['path'], ctx=ctx)
            sub = argparse.Namespace(
                seed=args.seed, quick=args.quick, warmup_cap=args.warmup_cap,
                probe_every=args.probe_every, warmup_window=args.warmup_window,
                skip_warmup=args.skip_warmup,
            )
            if args.quick:
                sub.warmup_cap = min(sub.warmup_cap, 1500)
            res = run_mod.run_e0(rig, sub, ctx, blend=False)
            outdir = os.path.join(args.out, e['name'])
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, 'summary.json'), 'w') as f:
                json.dump(res, f, indent=2)
            run_mod.plot_e0(res, outdir, e['name'])
            results[e['name']] = res
        except Exception:
            print(f"  FAILED: {traceback.format_exc()}", flush=True)

    # E0 aggregate. The full aggregate.csv from the spec needs E1-E3 columns and
    # lands when those experiments do.
    if results:
        agg = os.path.join(args.out, 'e0_aggregate.csv')
        with open(agg, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['config', 'particles_per_pixel', 'memory_time_steps',
                        'steps_per_sec', 'propagator_max_rel_err',
                        'warmup_steps', 'warmup_settled', 'warmup_in_memory_times'])
            for name, r in sorted(results.items()):
                wm = r.get('E0.4_warmup', {})
                w.writerow([
                    name,
                    round(r['meta']['particles_per_pixel'], 4),
                    round(r['meta']['memory_time_steps'], 2),
                    round(r['E0.1_throughput']['steps_per_sec'], 2),
                    f"{r['E0.3_propagator']['max_rel_error']:.3e}",
                    wm.get('warmup_steps', ''), wm.get('settled', ''),
                    round(wm.get('warmup_in_memory_times', 0), 1) if wm else '',
                ])
        print(f"\n-> {agg}")


if __name__ == '__main__':
    main()
