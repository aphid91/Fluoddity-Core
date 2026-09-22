"""E1: timescales.

The reduced model only pays off if fast and slow variables are actually
separated. E1 measures both sides on the live, warmed-up simulation:

  fast  -- how long a particle remembers its heading (tau_orient) and how long
           its motion stays ballistic before going diffusive (tau_cross)
  slow  -- how long the canvas pattern persists, per spatial-frequency band
           (tau_canvas)

and reports the ratio. A ratio near 1 means there is nothing to exploit.

EVEN LAGS ONLY. Every lag-indexed statistic here uses tau = 2, 4, 6, ...
A particle's velocity frequently reverses on consecutive steps -- not only
under negative drag, but whenever the force term outweighs the retained
momentum, which happens in ordinary configs too. On odd lags that reversal
shows up as strong anti-correlation and swamps the decay we are trying to
measure. Odd lags are still recorded, but only as a diagnostic (see
`odd_lag_diagnostic`), never as a reported timescale.
"""

import numpy as np

from . import propagator as prop

# Heading is undefined for a stationary particle; below this speed it is dropped.
MIN_SPEED = 1e-12


def record_tracked(rig, n_steps, blocks, stride=1):
    """Step the live sim, reading the tracked particles every step.

    Returns pos (T, N, 2) and vel (T, N, 2) in world units.
    """
    pos = np.empty((n_steps, sum(c for _, c in blocks), 2), dtype=np.float32)
    vel = np.empty_like(pos)
    for t in range(n_steps):
        rig.step(stride)
        e = rig.read_tracked(blocks)
        pos[t] = e[:, 0:2]
        vel[t] = e[:, 2:4]
    return pos, vel


def unwrap_displacements(pos):
    """Per-step displacements under the minimum-image convention.

    Position wraps on [-1, 1], so a particle crossing the seam looks like it
    jumped the full width. Minimum image takes the shorter of the two paths,
    which is correct as long as a single step is shorter than half the domain --
    true here by orders of magnitude.
    """
    dp = np.diff(pos, axis=0)
    return dp - 2.0 * np.round(dp / 2.0)


def detect_resets(dp, vel, strafe_power, sigma=20.0):
    """Mask of (step, particle) entries where a hazard reset teleported a particle.

    A normal step satisfies dpos = vel_new + strafe*strafe_power, so the residual
    dpos - vel_new is the strafe hop and is small. A reset re-places the particle
    somewhere unrelated to its velocity, leaving a residual far outside the
    strafe distribution. Particles flagged anywhere are dropped entirely rather
    than patched, so no window straddles a reset.
    """
    resid = np.linalg.norm(dp - vel[1:], axis=-1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-30
    bad = resid > med + sigma * 1.4826 * mad
    return bad.any(axis=0), resid


def heading_autocorrelation(vel, lags, t_stride=4):
    """<cos(theta_t - theta_{t+tau})> over particles and start times."""
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    u = np.where(speed > MIN_SPEED, vel / np.maximum(speed, MIN_SPEED), 0.0)
    good = (speed[..., 0] > MIN_SPEED)
    out = []
    T = u.shape[0]
    for lag in lags:
        if lag >= T:
            out.append(np.nan); continue
        a, b = u[:T - lag:t_stride], u[lag::t_stride]
        m = good[:T - lag:t_stride] & good[lag::t_stride]
        n = m.sum()
        out.append(float((a * b).sum(axis=-1)[m].sum() / n) if n else np.nan)
    return np.array(out)


def mean_squared_displacement(dp, lags, t_stride=4):
    """MSD(tau) from per-step displacements, in world units squared."""
    disp = np.cumsum(dp, axis=0)          # displacement from t=0, already unwrapped
    out = []
    T = disp.shape[0]
    for lag in lags:
        if lag >= T:
            out.append(np.nan); continue
        d = disp[lag::t_stride] - disp[:T - lag:t_stride]
        out.append(float((d ** 2).sum(axis=-1).mean()))
    return np.array(out)


def _decay_time(lags, acf, level=1.0 / np.e):
    """First lag where the autocorrelation drops below `level`, interpolated.

    Reported as a lag, i.e. in steps. Returns None if it never drops, which is
    itself informative: the heading has not decorrelated within the window.
    """
    for i in range(1, len(acf)):
        if np.isfinite(acf[i]) and acf[i] < level <= acf[i - 1]:
            f = (acf[i - 1] - level) / max(acf[i - 1] - acf[i], 1e-30)
            return float(lags[i - 1] + f * (lags[i] - lags[i - 1]))
    return None


def canvas_autocorrelation(rig, taus, n_bands=8):
    """Per-band correlation between the canvas now and the canvas tau steps later.

    Banding by |k| is what separates fast shot noise at high spatial frequency
    from the slow evolution of the pattern itself. A single whole-field
    correlation mixes the two and reports neither.
    """
    N = rig.canvas_dim
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.fftfreq(N)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    edges = np.linspace(0, kr.max() + 1e-12, n_bands + 1)
    masks = [(kr >= edges[i]) & (kr < edges[i + 1]) for i in range(n_bands)]

    def spec(c):
        return [np.fft.fft2(c[..., 0]), np.fft.fft2(c[..., 1])]

    c0 = rig.read_canvas()[..., :2].astype(np.float64)
    F0 = spec(c0)
    band_power0 = [float(sum((np.abs(F[m]) ** 2).sum() for F in F0)) for m in masks]

    rows, done = [], 0
    for tau in taus:
        rig.step(tau - done); done = tau
        Ft = spec(rig.read_canvas()[..., :2].astype(np.float64))
        corr = []
        for m in masks:
            num = sum(float(np.real(A[m] * np.conj(B[m])).sum()) for A, B in zip(F0, Ft))
            d0 = sum(float((np.abs(A[m]) ** 2).sum()) for A in F0)
            dt = sum(float((np.abs(B[m]) ** 2).sum()) for B in Ft)
            corr.append(num / max(np.sqrt(d0 * dt), 1e-30))
        rows.append({'tau': int(tau), 'corr_by_band': corr})
    return {
        'band_edges_cycles_per_px': [float(e) for e in edges],
        'band_power_at_t0': band_power0,
        # The band holding the most power is the pattern itself; that is the one
        # whose timescale the reduced model would have to respect.
        'pattern_band': int(np.argmax(band_power0)),
        'curves': rows,
    }


def run(rig, track_steps=1200, n_tracked=4096, max_lag=400,
        canvas_tau_max=8000, n_canvas_probes=18, n_bands=8, seed=0):
    px = rig.px_per_world
    strafe_power = float(rig.config['strafe_power'])

    # ---- Lagrangian statistics on tracked particles -------------------------
    blocks = rig.tracked_blocks(n_tracked=n_tracked, seed=seed)
    pos, vel = record_tracked(rig, track_steps, blocks)
    dp = unwrap_displacements(pos)
    reset_mask, resid = detect_resets(dp, vel, strafe_power)
    keep = ~reset_mask
    pos, vel, dp = pos[:, keep], vel[:, keep], dp[:, keep]

    even = np.arange(2, max_lag + 1, 2)
    odd = np.arange(1, max_lag + 1, 2)
    acf_even = heading_autocorrelation(vel, even)
    acf_odd = heading_autocorrelation(vel, odd)
    msd_even = mean_squared_displacement(dp, even)

    # Ballistic gives MSD ~ tau^2, diffusive ~ tau^1. The crossover is where the
    # local log-log slope first falls below 1.5.
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.gradient(np.log(np.maximum(msd_even, 1e-300)), np.log(even))
    tau_cross = None
    for i in range(1, len(even)):
        if np.isfinite(logs[i]) and logs[i] < 1.5:
            tau_cross = float(even[i]); break

    speed_px = np.linalg.norm(vel, axis=-1) * px
    strafe_px = np.linalg.norm(dp - vel[1:], axis=-1) * px

    tau_orient = _decay_time(even, acf_even)

    # ---- Eulerian statistics on the canvas ----------------------------------
    taus = np.unique(np.round(np.logspace(
        0, np.log10(canvas_tau_max), n_canvas_probes)).astype(int))
    canvas = canvas_autocorrelation(rig, taus, n_bands=n_bands)
    pb = canvas['pattern_band']
    pattern_curve = [(r['tau'], r['corr_by_band'][pb]) for r in canvas['curves']]
    tau_canvas = _decay_time(np.array([t for t, _ in pattern_curve]),
                             np.array([c for _, c in pattern_curve]))

    fast = max([v for v in (tau_orient, tau_cross) if v is not None], default=None)
    separation = (tau_canvas / fast) if (tau_canvas and fast) else None

    return {
        'lags_are_even_only': True,
        'why_even_only': ('velocity reverses on consecutive steps whenever force '
                          'outweighs retained momentum, not only under negative '
                          'drag; odd lags therefore measure that reversal rather '
                          'than heading decorrelation'),
        'tracked': {
            'requested': n_tracked,
            'kept_after_reset_filter': int(pos.shape[1]),
            'steps': track_steps,
            'strafe_residual_median_px': float(np.median(resid) * px),
        },
        'heading': {
            'lags': even.tolist(),
            'acf_even': [None if not np.isfinite(v) else float(v) for v in acf_even],
            'tau_orient_steps': tau_orient,
            'acf_at_lag_2': float(acf_even[0]),
        },
        # Diagnostic only. If acf_at_lag_1 is strongly negative the velocity is
        # flipping every step and odd lags are meaningless, which is the whole
        # reason nothing above uses them.
        'odd_lag_diagnostic': {
            'lags': odd.tolist(),
            'acf_odd': [None if not np.isfinite(v) else float(v) for v in acf_odd],
            'acf_at_lag_1': float(acf_odd[0]),
            'flip_detected': bool(acf_odd[0] < 0),
        },
        'msd': {
            'lags': even.tolist(),
            'msd_px2': (msd_even * px * px).tolist(),
            'tau_cross_steps': tau_cross,
            'diffusivity_px2_per_step': float(msd_even[-1] * px * px / (4.0 * even[-1])),
        },
        'motion_px_per_step': {
            'speed_mean': float(speed_px.mean()), 'speed_p99': float(np.percentile(speed_px, 99)),
            'strafe_mean': float(strafe_px.mean()), 'strafe_p99': float(np.percentile(strafe_px, 99)),
        },
        'canvas': canvas,
        'tau_canvas_pattern_steps': tau_canvas,
        'canvas_tau_max': int(canvas_tau_max),
        'separation_ratio': separation,
        'separation_note': (None if separation else
                            'tau_canvas did not decay to 1/e within canvas_tau_max; '
                            'the ratio is a lower bound of canvas_tau_max / fast'),
    }
