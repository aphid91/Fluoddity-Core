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
from . import e1 as e1_mod
from . import e2 as e2_mod
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
        pts = [t for t in w['trace'] if t.get('drift') is not None]
        allsteps = [t['step'] for t in w['trace']]
        power = [t['total_power'] for t in w['trace']]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        if pts:
            # Drift against its own noise floor: settling is where they cross,
            # so both curves have to be on the plot for it to mean anything.
            axes[0].semilogy([t['step'] for t in pts], [t['drift'] for t in pts],
                             '-', label='split-half drift')
            axes[0].semilogy([t['step'] for t in pts], [t['noise'] for t in pts],
                             '--', c='r', label='noise floor')
        if w['settled']:
            axes[0].axvline(w['warmup_steps'], ls=':', c='g', label='settled')
        axes[0].set_xlabel('step'); axes[0].set_ylabel('spectral drift')
        axes[0].legend(fontsize=8)
        axes[1].semilogy(allsteps, power, '-')
        axes[1].set_xlabel('step'); axes[1].set_ylabel('total canvas power')
        fig.suptitle(f'{name}: warm-up', fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'e0_warmup.png'), dpi=120)
        plt.close(fig)


def plot_e1(res, outdir, name):
    lags = np.array(res['heading']['lags'])
    acf = np.array([np.nan if v is None else v for v in res['heading']['acf_even']])
    odd_l = np.array(res['odd_lag_diagnostic']['lags'])
    odd = np.array([np.nan if v is None else v for v in res['odd_lag_diagnostic']['acf_odd']])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(lags, acf, '-', label='even lags (reported)')
    axes[0].plot(odd_l, odd, ':', c='0.6', label='odd lags (diagnostic)')
    axes[0].axhline(1 / np.e, ls='--', c='r', lw=0.8)
    axes[0].axhline(0, ls='-', c='0.8', lw=0.8)
    if res['heading']['tau_orient_steps']:
        axes[0].axvline(res['heading']['tau_orient_steps'], ls=':', c='g')
    axes[0].set_xlabel('lag (steps)'); axes[0].set_ylabel('heading autocorrelation')
    axes[0].legend(fontsize=7)

    m = np.array(res['msd']['msd_px2'])
    axes[1].loglog(lags, m, '-')
    if res['msd']['tau_cross_steps']:
        axes[1].axvline(res['msd']['tau_cross_steps'], ls=':', c='g', label='ballistic->diffusive')
        axes[1].legend(fontsize=7)
    axes[1].set_xlabel('lag (steps)'); axes[1].set_ylabel('MSD (px^2)')

    c = res['canvas']; pb = c['pattern_band']
    taus = [r['tau'] for r in c['curves']]
    for b in range(len(c['band_power_at_t0'])):
        vals = [r['corr_by_band'][b] for r in c['curves']]
        axes[2].semilogx(taus, vals, lw=2 if b == pb else 0.8,
                         label=f'band {b}' + (' (pattern)' if b == pb else ''))
    axes[2].axhline(1 / np.e, ls='--', c='r', lw=0.8)
    axes[2].set_xlabel('lag (steps)'); axes[2].set_ylabel('canvas correlation')
    axes[2].legend(fontsize=6, ncol=2)
    fig.suptitle(f'{name}: E1 timescales (even lags only)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'e1_timescales.png'), dpi=120)
    plt.close(fig)


def plot_e2(results, outdir, name):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for res, _ in results:
        i = res['snapshot_index']
        T = [r['T'] for r in res['convergence']]
        sh = [r['split_half_diff'] for r in res['convergence']]
        Ts = [t for t, v in zip(T, sh) if v is not None]
        vs = [v for v in sh if v is not None]
        axes[0].loglog(Ts, vs, '-o', ms=3, label=f'snapshot {i}')
        d = res['density_drift']
        axes[1].semilogx([x['T'] for x in d],
                         [x['density_corr_to_freeze'] for x in d], '-o', ms=3)
    axes[0].axhline(0.05, ls='--', c='r', lw=0.8, label='tol')
    axes[0].set_xlabel('frozen steps T'); axes[0].set_ylabel('split-half diff in m')
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.9, ls='--', c='r', lw=0.8)
    axes[1].set_xlabel('frozen steps T'); axes[1].set_ylabel('density corr to freeze time')
    fig.suptitle(f'{name}: E2 frozen canvas', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'e2_convergence.png'), dpi=120)
    plt.close(fig)

    # Field images from the first snapshot.
    _, f = results[0]
    def mag(v):
        return np.linalg.norm(v, axis=-1)
    panels = [('C* (frozen canvas)', mag(f['c_star'])),
              ('B-bar (mean deposit)', mag(f['b_bar'])),
              ('rho-bar (density)', f['rho']),
              ('|m| (deposit per particle)', mag(f['m'])),
              ('C_ss implied by B-bar', mag(f['c_ss'])),
              ('C_ss - C* residual', mag(f['c_ss'] - f['c_star']))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (title, img) in zip(axes.ravel(), panels):
        hi = np.quantile(img, 0.995) or 1.0
        ax.imshow(img, origin='lower', vmin=0, vmax=hi, cmap='magma')
        ax.set_title(title, fontsize=8); ax.axis('off')
    fig.suptitle(f'{name}: E2 fields', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'e2_fields.png'), dpi=110)
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
            rig, cap=args.warmup_cap, probe_every=args.probe_every, window=args.warmup_window
        )
        w = res['E0.4_warmup']
        print(f"       {'settled at' if w['settled'] else 'CAP HIT at'} "
              f"{w['warmup_steps']} steps ({w['wall_seconds']:.0f}s)", flush=True)
    return res


def run_e1_e2(rig, args, ctx, res, outdir, name):
    """E1 and E2 both need a settled canvas, so warm up once and share it.

    Warm-up always runs, even when a stored E0.4 result is present. That result
    came from a different process; this Rig is at frame 0, and a stored number
    does not put the live simulation into a settled state.
    """
    print(f'  warm-up (cap {args.warmup_cap}) ...', flush=True)
    res['E0.4_warmup'] = e0_mod.warmup(rig, cap=args.warmup_cap,
                                       probe_every=args.probe_every,
                                       window=args.warmup_window)
    w = res['E0.4_warmup']
    print(f"       {'settled' if w['settled'] else 'NOT settled'} at "
          f"{w['warmup_steps']} steps ({w['wall_seconds']:.0f}s)", flush=True)

    if 'e1' in args.experiments:
        print('  E1 timescales ...', flush=True)
        res['E1_timescales'] = e1_mod.run(
            rig, track_steps=args.track_steps, n_tracked=args.n_tracked,
            canvas_tau_max=args.canvas_tau_max, seed=args.seed)
        r = res['E1_timescales']
        print(f"       tau_orient={r['heading']['tau_orient_steps']} "
              f"tau_cross={r['msd']['tau_cross_steps']} "
              f"tau_canvas={r['tau_canvas_pattern_steps']} "
              f"separation={r['separation_ratio']}", flush=True)
        plot_e1(r, outdir, name)

    if 'e2' in args.experiments:
        print(f'  E2 frozen canvas ({args.n_snapshots} snapshots) ...', flush=True)
        pairs = e2_mod.run(rig, n_snapshots=args.n_snapshots,
                           steps_between=args.steps_between, t_max=args.frozen_steps)
        res['E2_frozen'] = [r for r, _ in pairs]
        for r in res['E2_frozen']:
            print(f"       snap {r['snapshot_index']}: T_conv={r['T_conv']} "
                  f"usable_window={r['usable_window_exists']} "
                  f"stationarity_resid={r['stationarity_residual']:.3f}", flush=True)
        plot_e2(pairs, outdir, name)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true', help='short runs for smoke testing')
    ap.add_argument('--warmup-cap', type=int, default=30000)
    ap.add_argument('--probe-every', type=int, default=100)
    ap.add_argument('--warmup-window', type=int, default=20,
                    help='probes in the trailing split-half drift window')
    ap.add_argument('--skip-warmup', action='store_true')
    ap.add_argument('--experiments', default='e0',
                    help='comma-separated: e0,e1,e2')
    ap.add_argument('--track-steps', type=int, default=1200)
    ap.add_argument('--n-tracked', type=int, default=4096)
    ap.add_argument('--canvas-tau-max', type=int, default=20000)
    ap.add_argument('--n-snapshots', type=int, default=3)
    ap.add_argument('--steps-between', type=int, default=1500)
    ap.add_argument('--frozen-steps', type=int, default=2000)
    args = ap.parse_args(argv)
    args.experiments = [x.strip() for x in args.experiments.split(',') if x.strip()]

    if args.quick:
        args.warmup_cap = min(args.warmup_cap, 1500)

    ctx = create_context()
    rig = Rig(args.config, ctx=ctx)
    name = os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join(args.out, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"[{name}] persistence={rig.persistence:.3f} "
          f"memory={rig.memory_time:.1f} steps cohorts={rig.config['cohorts']}", flush=True)
    # Merge into any existing summary so running E1/E2 does not discard the E0
    # results already measured for this config.
    summary_path = os.path.join(outdir, 'summary.json')
    res = {}
    if os.path.exists(summary_path):
        try:
            res = json.load(open(summary_path))
        except Exception as e:
            print(f'  existing summary unreadable, starting fresh: {e}')
    res['meta'] = metadata(rig, args, ctx)
    if 'e0' in args.experiments:
        res.update(run_e0(rig, args, ctx))
    if 'e1' in args.experiments or 'e2' in args.experiments:
        res = run_e1_e2(rig, args, ctx, res, outdir, name)
    with open(summary_path, 'w') as f:
        json.dump(res, f, indent=2)
    # E0.2 is config-independent, so it also lands at the top level where the
    # report picks it up once rather than per config.
    if 'E0.2_blend' in res:
        with open(os.path.join(args.out, 'e0_blend_check.json'), 'w') as f:
            json.dump(res['E0.2_blend'], f, indent=2)
    plot_e0(res, outdir, name)
    print(f"[{name}] -> {outdir}/summary.json", flush=True)
    return res


if __name__ == '__main__':
    main()
