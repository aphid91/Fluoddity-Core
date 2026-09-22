"""Run the rig's experiments on a single config.

    python -m closure_rig.run --config physics_configs/Pop.json --out results/

Only E0 is implemented so far; E1-E3 land on the same CLI once E0's findings
are signed off (spec section 8.2 makes E0 a hard checkpoint).
"""

import argparse
import hashlib
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import e0 as e0_mod
from .harness import Rig, create_context, git_commit, gpu_info


def config_hash(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]


def metadata(rig, args, ctx):
    """Everything needed to tell two runs apart later."""
    info = gpu_info(ctx)
    return {
        'config_path': rig.config_path,
        'config_name': os.path.splitext(os.path.basename(rig.config_path))[0],
        'config_md5': config_hash(rig.config_path),
        'git_commit': git_commit(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'gpu': info,
        # Loud, because a throughput number from llvmpipe must never be used as
        # the cost baseline for a speedup claim.
        'throughput_is_comparable': not info['software_rendered'],
        'seed': args.seed,
        'quick': args.quick,
        'particles': rig.system.entity_buffer.size // 24,
        'canvas_dim': rig.canvas_dim,
        'particles_per_pixel': rig.particles_per_pixel,
        'persistence_used': rig.persistence,
        'memory_time_steps': rig.memory_time,
        'px_per_world_unit': rig.px_per_world,
    }


def plot_e0(res, outdir, name):
    """Two plots: does the propagator hold, and did the canvas settle."""
    p = res.get('E0.3_propagator')
    if p:
        ks = [r['k'] for r in p['results']]
        errs = [r['rel_l2_error'] for r in p['results']]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.loglog(ks, errs, 'o-')
        ax.axhline(1e-6, ls='--', c='0.6', label='float32 noise floor (~1e-6)')
        ax.set_xlabel('steps k'); ax.set_ylabel('relative L2 error')
        ax.set_title(f'{name}: shader vs closed-form propagator')
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'e0_propagator_error.png'), dpi=120)
        plt.close(fig)

    w = res.get('E0.4_warmup')
    if w and w['trace']:
        steps = [t['step'] for t in w['trace'] if t['delta'] is not None]
        deltas = [t['delta'] for t in w['trace'] if t['delta'] is not None]
        power = [t['total_power'] for t in w['trace']]
        allsteps = [t['step'] for t in w['trace']]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].semilogy(steps, deltas, '-')
        axes[0].axhline(w['tol'], ls='--', c='r', label=f"tol={w['tol']}")
        if w['settled']:
            axes[0].axvline(w['warmup_steps'], ls=':', c='g', label='settled')
        axes[0].set_xlabel('step'); axes[0].set_ylabel('spectral change')
        axes[0].legend(fontsize=8)
        axes[1].semilogy(allsteps, power, '-')
        axes[1].set_xlabel('step'); axes[1].set_ylabel('total canvas power')
        fig.suptitle(f'{name}: warm-up', fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'e0_warmup.png'), dpi=120)
        plt.close(fig)


def run_e0(rig, args, ctx, blend=True):
    """E0 for one config. blend=False skips E0.2, which is config-independent
    and is written once as results/e0_blend_check.json by the batch runner."""
    res = {'meta': metadata(rig, args, ctx)}
    print('  E0.1 throughput ...', flush=True)
    res['E0.1_throughput'] = e0_mod.throughput(rig, steps=10 if args.quick else 50)
    print(f"       {res['E0.1_throughput']['steps_per_sec']:.2f} steps/sec", flush=True)

    if blend:
        print('  E0.2 blend check ...', flush=True)
        res['E0.2_blend'] = e0_mod.blend_check(rig)

    print('  E0.3 propagator ...', flush=True)
    ks = (1, 10) if args.quick else (1, 10, 100, 1000)
    res['E0.3_propagator'] = e0_mod.propagator_test(rig, ks=ks, seed=args.seed)
    print(f"       max rel err {res['E0.3_propagator']['max_rel_error']:.3e}", flush=True)

    if not args.skip_warmup:
        print(f'  E0.4 warm-up (cap {args.warmup_cap}) ...', flush=True)
        res['E0.4_warmup'] = e0_mod.warmup(
            rig, cap=args.warmup_cap, probe_every=args.probe_every, tol=args.warmup_tol
        )
        w = res['E0.4_warmup']
        print(f"       {'settled at' if w['settled'] else 'CAP HIT at'} "
              f"{w['warmup_steps']} steps ({w['wall_seconds']:.0f}s)", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true', help='short runs for smoke testing')
    ap.add_argument('--warmup-cap', type=int, default=30000)
    ap.add_argument('--probe-every', type=int, default=250)
    ap.add_argument('--warmup-tol', type=float, default=0.02)
    ap.add_argument('--skip-warmup', action='store_true')
    args = ap.parse_args(argv)

    if args.quick:
        args.warmup_cap = min(args.warmup_cap, 1500)

    ctx = create_context()
    rig = Rig(args.config, ctx=ctx)
    name = os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join(args.out, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"[{name}] persistence={rig.persistence:.3f} "
          f"memory={rig.memory_time:.1f} steps cohorts={rig.config['cohorts']}", flush=True)
    res = run_e0(rig, args, ctx)
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=2)
    # E0.2 is config-independent, so it also lands at the top level where the
    # report picks it up once rather than per config.
    with open(os.path.join(args.out, 'e0_blend_check.json'), 'w') as f:
        json.dump(res['E0.2_blend'], f, indent=2)
    plot_e0(res, outdir, name)
    print(f"[{name}] -> {outdir}/summary.json", flush=True)
    return res


if __name__ == '__main__':
    main()
