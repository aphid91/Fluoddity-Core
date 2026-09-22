"""Build results/E0_REPORT.md from the per-config summaries.

    python -m closure_rig.report --out results/
"""

import argparse
import csv
import glob
import json
import os


def load(out):
    runs = {}
    for p in sorted(glob.glob(os.path.join(out, '*', 'summary.json'))):
        try:
            runs[os.path.basename(os.path.dirname(p))] = json.load(open(p))
        except Exception as e:
            print(f"  unreadable: {p}: {e}")
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args(argv)
    runs = load(args.out)
    if not runs:
        print('no summaries found'); return

    if not args.no_plots:
        from .run import plot_e0
        for name, r in runs.items():
            try:
                plot_e0(r, os.path.join(args.out, name), name)
            except Exception as e:
                print(f'  plot failed for {name}: {e}')

    # Aggregate is rebuilt here rather than only inside run_all, so it survives
    # a crash partway through a sweep.
    with open(os.path.join(args.out, 'e0_aggregate.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['config', 'particles_per_pixel', 'memory_time_steps', 'steps_per_sec',
                     'propagator_max_rel_err', 'warmup_steps', 'warmup_settled',
                     'warmup_in_memory_times'])
        for name, r in sorted(runs.items()):
            wm = r.get('E0.4_warmup', {})
            wr.writerow([
                name, round(r['meta']['particles_per_pixel'], 4),
                round(r['meta']['memory_time_steps'], 2),
                round(r['E0.1_throughput']['steps_per_sec'], 2),
                f"{r['E0.3_propagator']['max_rel_error']:.3e}",
                wm.get('warmup_steps', ''), wm.get('settled', ''),
                round(wm.get('warmup_in_memory_times', 0), 1) if wm else '',
            ])

    any_meta = next(iter(runs.values()))['meta']
    soft = any_meta['gpu']['software_rendered']
    blend_path = os.path.join(args.out, 'e0_blend_check.json')
    blend = json.load(open(blend_path)) if os.path.exists(blend_path) else None

    L = []
    L.append('# E0: sanity and calibration\n')
    L.append(f"Configs measured: **{len(runs)}**. "
             f"Renderer: `{any_meta['gpu']['GL_RENDERER']}`, "
             f"GL `{any_meta['gpu']['GL_VERSION']}`.\n")
    if soft:
        L.append('> **Software-rendered run.** Throughput below is CPU rasterisation '
                 'and must not be used as the cost baseline for any speedup claim. '
                 'Re-run E0.1 on a GPU for that number. Every other E0 result is '
                 'renderer-independent.\n')

    if blend:
        sp = blend['subpixel_sweep']
        s = blend['single_particle']
        L.append('## E0.2 Blend check\n')
        L.append(f"- `brush.xy = sum(vel * k^2)`, `brush.w = sum(k^2)`, "
                 f"`brush.z = sum(0.01 * k^2)`.\n")
        L.append(f"- Velocity recovered from `brush.xy / brush.w`: "
                 f"**exact** (max deviation {sp['max_vel_deviation']:.1e} across all offsets).\n")
        L.append(f"- Splat quad is {blend['quad_px']:.2f} px wide; the Gaussian is "
                 f"sigma = {blend['sigma_px']:.2f} px, point-sampled at "
                 f"{sp['min_pixels_touched']}-{sp['max_pixels_touched']} pixel centres.\n")
        L.append(f"- Deposit per particle varies **{sp['max_over_min']:.0f}x** with sub-pixel "
                 f"position (CV {sp['cv_percent']:.0f}%), so `brush.w` carries "
                 f"{100/sp['variance_inflation']:.0f}% of the information of a true count.\n")
        L.append(f"- This test is config-independent (no config uniform reaches brush.frag), "
                 f"so it is reported once rather than per config.\n")

    L.append('\n## E0.3 Propagator\n')
    worst = max(runs.items(), key=lambda kv: kv[1]['E0.3_propagator']['max_rel_error'])
    best = min(runs.items(), key=lambda kv: kv[1]['E0.3_propagator']['max_rel_error'])
    L.append(f"Shader vs closed-form Fourier update, channels `.xy`, k up to 1000.\n\n")
    L.append(f"- Worst config: **{worst[0]}**, relative L2 error "
             f"{worst[1]['E0.3_propagator']['max_rel_error']:.2e}\n")
    L.append(f"- Best config: **{best[0]}**, {best[1]['E0.3_propagator']['max_rel_error']:.2e}\n")

    warm = {n: r['E0.4_warmup'] for n, r in runs.items() if 'E0.4_warmup' in r}
    if warm:
        L.append('\n## E0.4 Warm-up\n')
        L.append('| config | memory time | warm-up steps | in memory times | evals | verdict |')
        L.append('|---|---|---|---|---|---|')
        for n, w in sorted(warm.items(), key=lambda kv: kv[1]['memory_time_steps']):
            verdict = ('**floor-limited**' if w.get('floor_limited')
                       else ('settled' if w['settled'] else '**cap hit**'))
            L.append(f"| {n} | {w['memory_time_steps']:.1f} | {w['warmup_steps']} | "
                     f"{w['warmup_in_memory_times']:.0f} | "
                     f"{w.get('evaluations_before_firing', '-')} | {verdict} |")
        L.append('\nFloor-limited means the drift test passed on its first evaluation, '
                 'so the figure is the earliest step the window can credit rather than a '
                 'measured settling time.\n')

    L.append('\n## Per-config table\n')
    L.append('| config | particles/px | memory time (steps) | steps/sec | propagator max rel err | warm-up |')
    L.append('|---|---|---|---|---|---|')
    for name, r in sorted(runs.items()):
        w = r.get('E0.4_warmup')
        wcol = '-'
        if w:
            wcol = (f"{w['warmup_steps']} (floor-limited)" if w.get('floor_limited')
                    else (str(w['warmup_steps']) if w['settled'] else 'cap hit'))
        L.append(f"| {name} | {r['meta']['particles_per_pixel']:.2f} | "
                 f"{r['meta']['memory_time_steps']:.1f} | "
                 f"{r['E0.1_throughput']['steps_per_sec']:.1f} | "
                 f"{r['E0.3_propagator']['max_rel_error']:.1e} | {wcol} |")

    path = os.path.join(args.out, 'E0_REPORT.md')
    open(path, 'w').write('\n'.join(L) + '\n')
    print(f'-> {path}')


if __name__ == '__main__':
    main()
