"""E2: frozen-canvas test. The main experiment.

The canvas update is exactly linear and shift-invariant (verified in E0.3), so a
reduced model can advance the canvas any number of steps in one Fourier solve.
The only thing left to model is B-bar, the time-averaged momentum deposit. E2
asks whether B-bar is a well-defined object at all:

  1. Freeze the canvas at C*. Particles keep moving and depositing, but the
     field they sense never changes.
  2. Does m = B-bar / rho-bar converge, and after how many steps (T_conv)?
  3. Freezing lets particles migrate where the real feedback would never send
     them, so rho drifts and the measurement goes stale. Is there a window where
     m has converged but rho has not yet drifted? If not, there is no valid
     frozen measurement and the whole approach fails here.
  4. Does the canvas implied by the measured deposit match the one we froze?
     C_ss = (1-p)(I - p*Blur)^-1 B-bar, compared against C*. A small residual
     means the pattern is in quasi-equilibrium with its own deposit, which is
     exactly what the reduced model assumes.

Note on noise: E0.2 found the deposit is kernel-squared weighted and severely
aliased by sub-pixel position (CV 124%), inflating variance on any density built
from brush.w by 2.54x. T_conv here is correspondingly longer than a naive
Poisson estimate would suggest. m itself is unaffected in the mean, because the
aliasing cancels in the B-bar / rho-bar ratio.
"""

import numpy as np

from . import propagator as prop


def _dyadic_probes(t_max, n):
    """Probe times that halve exactly: t_max, t_max/2, t_max/4, ...

    Log spacing would be the obvious choice, but the split-half test at T needs
    the accumulator at T/2 as well, and with log spacing T/2 almost never lands
    on another probe -- most split-half points come back empty. Exact halving
    guarantees every probe except the smallest has its own half available.
    """
    out, t = [], int(t_max)
    for _ in range(n):
        out.append(t)
        if t <= 1:
            break
        t //= 2
    return np.array(sorted(set(out)))


def _rel_diff(a, b, mask):
    """Normalised difference between two vector fields over a mask."""
    if mask.sum() == 0:
        return float('nan')
    d = np.linalg.norm(a[mask] - b[mask], axis=-1)
    s = np.linalg.norm(a[mask], axis=-1) + np.linalg.norm(b[mask], axis=-1)
    return float(2.0 * d.sum() / max(s.sum(), 1e-30))


def frozen_window(rig, t_max=2000, n_probes=14, density_quantile=0.5):
    """Freeze the canvas and watch the deposit accumulate.

    Returns the convergence trace plus the stationarity residual.
    """
    c_star = rig.read_canvas()[..., :2].astype(np.float64)
    rig.reset_accumulator()

    probes = _dyadic_probes(t_max, n_probes)
    snaps, done = {}, 0
    for t in probes:
        rig.step_frozen(int(t - done)); done = int(t)
        a = rig.read_accumulator(mean=False).astype(np.float64)
        snaps[int(t)] = (a[..., :2].copy(), a[..., 3].copy())

    # Density threshold from the fully accumulated field, so every probe is
    # compared on the same set of pixels.
    full_xy, full_w = snaps[int(probes[-1])]
    thresh = float(np.quantile(full_w, density_quantile))
    mask = full_w > thresh
    m_final = np.zeros_like(full_xy)
    m_final[mask] = full_xy[mask] / full_w[mask][:, None]

    rows = []
    half_key = None
    for t in probes:
        xy, w = snaps[int(t)]
        m = np.zeros_like(xy)
        ok = mask & (w > 0)
        m[ok] = xy[ok] / w[ok][:, None]
        # Split-half: first half of the window vs the second half. This is what
        # separates "still drifting" from "just noisy" -- a purely noisy estimate
        # has the two halves agreeing in the mean.
        half = int(t) // 2
        split = None
        if half in snaps and half >= 1:
            xy1, w1 = snaps[half]
            xy2, w2 = xy - xy1, w - w1
            m1 = np.zeros_like(xy); m2 = np.zeros_like(xy)
            o1 = mask & (w1 > 0); o2 = mask & (w2 > 0)
            m1[o1] = xy1[o1] / w1[o1][:, None]
            m2[o2] = xy2[o2] / w2[o2][:, None]
            split = _rel_diff(m1, m2, mask & o1 & o2)
        rows.append({
            'T': int(t),
            'rel_diff_to_final': _rel_diff(m, m_final, ok),
            'split_half_diff': split,
            'mean_density': float(w[mask].mean() / max(t, 1)),
        })

    # --- density drift under the frozen field --------------------------------
    # Freezing lets particles migrate where the real feedback would never send
    # them, so the density the deposit is being divided by slowly stops being
    # the density at freeze time. The reference is averaged over the first
    # ref_steps rather than a single step, which would be far too noisy to
    # correlate against.
    drift = []
    ref_target = max(8, int(probes[-1]) // 16)
    ref_t = int(min((t for t in probes if t >= ref_target), default=probes[-1]))
    ref = snaps[ref_t][1] / float(ref_t)
    prev_t, prev_w = ref_t, snaps[ref_t][1]
    for t in probes:
        if int(t) <= ref_t:
            continue
        w = snaps[int(t)][1]
        cur = (w - prev_w) / max(int(t) - prev_t, 1)
        a, b = ref[mask], cur[mask]
        c = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float('nan')
        drift.append({'T': int(t), 'density_corr_to_freeze': c})
        prev_t, prev_w = int(t), w

    # --- stationarity residual: does C* match the canvas its own deposit implies?
    b_bar = full_xy / float(probes[-1])
    c_ss = prop.steady_state(b_bar, rig.config['trail_persistence'],
                             rig.config['trail_diffusion'])
    num = np.linalg.norm(c_ss - c_star)
    den = np.linalg.norm(c_star)
    resid_by_band = []
    N = rig.canvas_dim
    ky = np.fft.fftfreq(N)[:, None]; kx = np.fft.fftfreq(N)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    edges = np.linspace(0, kr.max() + 1e-12, 9)
    for i in range(8):
        bm = (kr >= edges[i]) & (kr < edges[i + 1])
        e = sum(float((np.abs(np.fft.fft2(c_ss[..., c] - c_star[..., c])[bm]) ** 2).sum())
                for c in range(2))
        r = sum(float((np.abs(np.fft.fft2(c_star[..., c])[bm]) ** 2).sum()) for c in range(2))
        resid_by_band.append(float(np.sqrt(e / max(r, 1e-30))))

    return {
        'T_max': int(probes[-1]),
        'density_reference_steps': ref_t,
        'density_quantile': density_quantile,
        'pixels_above_threshold': int(mask.sum()),
        'convergence': rows,
        'density_drift': drift,
        'stationarity_residual': float(num / max(den, 1e-30)),
        'stationarity_residual_by_band': resid_by_band,
        'fields': {'c_star': c_star, 'b_bar': b_bar, 'rho': full_w / float(probes[-1]),
                   'm': m_final, 'c_ss': c_ss, 'mask': mask},
    }


def run(rig, n_snapshots=3, steps_between=1500, t_max=2000, n_probes=14,
        tol=0.05):
    """Repeat the frozen-window measurement from several independent snapshots."""
    out = []
    for i in range(n_snapshots):
        if i > 0:
            rig.step(steps_between)      # decorrelate from the previous snapshot
        snap = rig.snapshot()
        res = frozen_window(rig, t_max=t_max, n_probes=n_probes)
        fields = res.pop('fields')
        # T_conv: the earliest probe whose split-half halves agree to `tol`,
        # i.e. the deposit has stopped drifting rather than merely being noisy.
        t_conv = None
        for r in res['convergence']:
            if r['split_half_diff'] is not None and r['split_half_diff'] < tol:
                t_conv = r['T']; break
        res['T_conv'] = t_conv
        res['T_conv_in_memory_times'] = (t_conv / rig.memory_time) if t_conv else None
        # A usable frozen window needs m converged while rho is still faithful.
        last_corr = res['density_drift'][-1]['density_corr_to_freeze'] if res['density_drift'] else None
        corr_at_conv = next((d['density_corr_to_freeze'] for d in res['density_drift']
                             if t_conv and d['T'] >= t_conv), None)
        res['density_corr_at_T_conv'] = corr_at_conv
        res['density_corr_at_T_max'] = last_corr
        res['usable_window_exists'] = bool(t_conv is not None and corr_at_conv is not None
                                           and corr_at_conv > 0.9)
        res['snapshot_index'] = i
        out.append((res, fields))
        rig.restore(snap)                # undo the frozen excursion
    return out
