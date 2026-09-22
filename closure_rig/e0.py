"""E0: sanity and calibration.

Nothing downstream is trustworthy until these four pass, which is why the spec
makes E0 a checkpoint:

  E0.1 throughput  -- the cost baseline every speedup claim is measured against.
  E0.2 blend check -- what actually lands in the brush texture. The deposit
                      normalisation and the density proxy both depend on it.
  E0.3 propagator  -- does the canvas really obey the closed-form Fourier update?
                      The whole premise of taking large timesteps rests on this.
  E0.4 warm-up     -- how many steps until canvas statistics stop changing, so
                      later experiments measure a settled system.
"""

import time

import numpy as np

from particle_system import ENTITY_COUNT, SIZE_OF_ENTITY_STRUCT
from . import propagator as prop
from .harness import FLOATS_PER_ENTITY

DEFAULT_SIZE = 0.0015 / 0.5      # entity_update.glsl: .0015 / SQRT_WORLD_SIZE


# ---------------------------------------------------------------- E0.1
def throughput(rig, steps=50):
    """Steps/sec for the unmodified sim with no readbacks."""
    sps = rig.benchmark(steps=steps)
    return {
        'steps_per_sec': sps,
        'steps_measured': steps,
        'particles': ENTITY_COUNT,
        'canvas_dim': rig.canvas_dim,
        'particles_per_pixel': rig.particles_per_pixel,
    }


# ---------------------------------------------------------------- E0.2
def blend_check(rig, grid=16, spacing=32):
    """Splat particles with known velocity and see what the texture holds.

    brush.frag writes vec4(vel, 0.01, 1.0) * kernel, and the blend is
    (SRC_ALPHA, ONE) with src.a = kernel, so the framebuffer accumulates
    vel*kernel^2 in .xy and kernel^2 in .w. The deposit is kernel-SQUARED
    weighted and .w is the natural density proxy.

    The catch this measures: the splat quad is ~1.5 px across and the Gaussian
    inside it has sigma ~0.25 px, so it is point-sampled at only a handful of
    pixel centres. How much kernel^2 a particle contributes therefore depends
    strongly on where it sits within its pixel. The sweep below walks a
    grid x grid lattice of sub-pixel offsets and quantifies that spread,
    because it sets the noise floor on any density estimate built from brush.w
    and hence on E2's convergence time.

    Nothing here depends on the config: no config uniform reaches brush.frag,
    and the splat size comes from a #define. Run it once, not once per config.
    """
    N = rig.canvas_dim
    vel = np.array([0.004, -0.002], dtype=np.float32)

    def splat(rows):
        ents = np.zeros((ENTITY_COUNT, FLOATS_PER_ENTITY), dtype=np.float32)
        for i, r in enumerate(rows):
            ents[i] = r
        rig.system.entity_buffer.write(ents.tobytes())
        rig.system.frame_count = 1   # brush.frag discards when frame_count == 0
        rig.system.create_brush()
        rig.ctx.finish()
        return rig.read_brush()

    # --- one isolated particle: the exact deposit relationship
    b = splat([(0.0, 0.0, vel[0], vel[1], DEFAULT_SIZE, 0.0)])
    s = b.reshape(-1, 4).sum(axis=0)
    single = {
        'sum_xy': [float(s[0]), float(s[1])],
        'sum_z': float(s[2]),
        'sum_w': float(s[3]),
        'recovered_vel': [float(s[0] / s[3]), float(s[1] / s[3])],
        'true_vel': [float(vel[0]), float(vel[1])],
        'vel_recovered_exactly': bool(np.allclose(s[:2] / s[3], vel, rtol=1e-4)),
        'z_over_w': float(s[2] / s[3]),      # the 0.01 literal in brush.frag
        'footprint_pixels': int((b[..., 3] > 0).sum()),
    }

    # --- deterministic sub-pixel sweep. grid*spacing must tile the canvas
    # exactly, or probes fall off the edge and read back as empty windows.
    if grid * spacing != N:
        spacing = N // grid
    r = spacing // 2 - 4                      # window well inside the cell
    rows, cells = [], []
    for i in range(grid * grid):
        px, py = spacing // 2 + (i % grid) * spacing, spacing // 2 + (i // grid) * spacing
        fx, fy = (i % grid) / grid, (i // grid) / grid
        rows.append((((px + fx) / N) * 2 - 1, ((py + fy) / N) * 2 - 1,
                     vel[0], vel[1], DEFAULT_SIZE, 0.0))
        cells.append((px, py))
    b = splat(rows)

    def window(px, py, c):
        return b[py - r:py + r + 1, px - r:px + r + 1, c]

    mass = np.array([window(px, py, 3).sum() for px, py in cells])
    vx = np.array([window(px, py, 0).sum() for px, py in cells])
    npix = np.array([(window(px, py, 3) > 0).sum() for px, py in cells])
    cv = float(mass.std() / mass.mean())
    quad_px = DEFAULT_SIZE * rig.px_per_world
    return {
        'quad_px': float(2 * quad_px),
        'sigma_px': float(0.163 * 2 * quad_px),
        'single_particle': single,
        'subpixel_sweep': {
            'grid': grid,
            'n_probes': int(mass.size),
            # If the per-cell windows recover the global total, the splats really
            # are isolated and additive blending does what we assume.
            'windows_captured_all': bool(np.isclose(mass.sum(), b[..., 3].sum(), rtol=1e-4)),
            'min_kernel_sq': float(mass.min()),
            'median_kernel_sq': float(np.median(mass)),
            'mean_kernel_sq': float(mass.mean()),
            'max_kernel_sq': float(mass.max()),
            'max_over_min': float(mass.max() / max(mass.min(), 1e-30)),
            'cv_percent': 100.0 * cv,
            'min_pixels_touched': int(npix.min()),
            'max_pixels_touched': int(npix.max()),
            # m = brush.xy/brush.w is a ratio, so the kernel^2 weighting cancels
            # and velocity comes back exactly however badly the mass is aliased.
            'max_vel_deviation': float(np.abs(vx / mass - vel[0]).max()),
            # Density built on brush.w is noisier than a count by this factor.
            'variance_inflation': float(1.0 + cv ** 2),
            'effective_sample_fraction': float(1.0 / (1.0 + cv ** 2)),
        },
        'interpretation': (
            'brush.xy = sum(vel*k^2); brush.w = sum(k^2); brush.z = sum(0.01*k^2). '
            'm = brush.xy/brush.w is exact. brush.w is proportional to particle '
            'count only up to the sub-pixel aliasing reported above, so a density '
            'estimate from it carries variance_inflation times the variance of a '
            'true count. Config-independent.'
        ),
    }


# ---------------------------------------------------------------- E0.3
def propagator_test(rig, ks=(1, 10, 100, 1000), seed=0):
    """Random canvas + fixed synthetic brush, shader vs closed form.

    Only .xy is compared: canvas.frag substitutes constants for the brush's .z
    and .w, so those channels are not driven by the brush and are outside the
    linear system the reduced model would advance.
    """
    N = rig.canvas_dim
    rng = np.random.default_rng(seed)
    p = rig.config['trail_persistence']
    td = rig.config['trail_diffusion']

    c0 = np.zeros((N, N, 4), dtype=np.float32)
    c0[..., :2] = rng.standard_normal((N, N, 2)).astype(np.float32) * 0.01
    brush = np.zeros((N, N, 4), dtype=np.float32)
    # Smooth, structured source: a constant plus a couple of low modes, so the
    # test exercises several Fourier bands rather than just DC.
    yy, xx = np.mgrid[0:N, 0:N] / float(N)
    brush[..., 0] = (0.002 + 0.001 * np.sin(2 * np.pi * 3 * xx)).astype(np.float32)
    brush[..., 1] = (-0.001 + 0.001 * np.cos(2 * np.pi * 5 * yy)).astype(np.float32)

    rows = []
    for k in ks:
        rig.write_canvas(c0)
        rig.write_brush(brush)
        rig.system.frame_count = 1        # anything but 0, which would wipe the canvas
        rig.step_canvas_only(k)
        rig.ctx.finish()
        got = rig.read_canvas()[..., :2].astype(np.float64)
        want = prop.propagate(c0[..., :2].astype(np.float64),
                              brush[..., :2].astype(np.float64), p, td, k)
        err = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30)
        rows.append({
            'k': k,
            'rel_l2_error': float(err),
            'max_abs_error': float(np.abs(got - want).max()),
            'field_rms': float(np.sqrt((want ** 2).mean())),
        })
    return {
        'persistence_used': prop.persistence(p),
        'trail_diffusion': float(td),
        'K': prop.diffusion_constant(td),
        'channels_compared': 'xy only (canvas.frag forces z=0, w=1)',
        'results': rows,
        'max_rel_error': max(r['rel_l2_error'] for r in rows),
    }


# ---------------------------------------------------------------- E0.4
def warmup(rig, cap=30000, probe_every=100, window=20, n_bands=16):
    """Run from reset until the canvas's radial power spectrum stops drifting.

    The statistic is translation-invariant on purpose: what matters is that the
    pattern has reached its characteristic scales, not where the filaments sit.

    Settling is decided by a split-half drift test over a trailing window of
    spectra, not by comparing consecutive probes. Consecutive probes differ
    mostly by fluctuation -- in a stationary state that difference has a noise
    floor well above any fixed tolerance one would pick, so a consecutive-probe
    test never fires no matter how settled the system is. Instead the window is
    split in half and the systematic change between halves is compared against
    the probe-to-probe scatter within them, so the threshold calibrates itself
    to however noisy the config happens to be. This is the same first-half /
    second-half separation of drift from noise that E2 uses.

    Returns the step at which drift fell to the noise level, or the cap with
    settled=False. A config that genuinely never settles is reported as such
    rather than having the threshold loosened until it passes.
    """
    rig.reset()
    rig.step(1)                            # step 1 wipes the canvas and places particles

    specs, trace, settled_at = [], [], None
    steps = 0
    half = window // 2
    t0 = time.perf_counter()
    while steps < cap:
        rig.step(probe_every)
        steps += probe_every
        spec, _ = prop.radial_power_spectrum(rig.read_canvas()[..., :2], n_bands=n_bands)
        specs.append(spec)
        if len(specs) > window:
            specs.pop(0)

        if len(specs) < window:
            trace.append({'step': steps, 'drift': None, 'noise': None,
                          'total_power': float(spec.sum())})
            continue

        arr = np.array(specs)
        m1, m2 = arr[:half].mean(axis=0), arr[half:].mean(axis=0)
        total = max(arr.mean(axis=0).sum(), 1e-30)
        # Systematic change between the two halves of the window.
        drift = float(np.abs(m1 - m2).sum() / total)
        # Scatter of a half-window mean, from the spread inside each half. The
        # difference of two independent half-means carries sqrt(2) times that.
        se = (np.abs(arr[:half] - m1).sum(axis=1).mean()
              + np.abs(arr[half:] - m2).sum(axis=1).mean()) / (2.0 * total * np.sqrt(half))
        noise = float(np.sqrt(2.0) * se)
        trace.append({'step': steps, 'drift': drift, 'noise': noise,
                      'total_power': float(arr.mean(axis=0).sum())})
        if drift <= noise:
            # The window is centred on the transition, so credit the midpoint.
            settled_at = steps - half * probe_every
            break

    return {
        'settled': settled_at is not None,
        'warmup_steps': settled_at if settled_at is not None else steps,
        'steps_run': steps,
        'cap': cap,
        'probe_every': probe_every,
        'window_probes': window,
        'criterion': 'split-half spectral drift <= sqrt(2) * within-window scatter',
        'memory_time_steps': rig.memory_time,
        'warmup_in_memory_times': (settled_at or steps) / rig.memory_time,
        'final_drift': trace[-1]['drift'] if trace and trace[-1]['drift'] is not None else None,
        'final_noise': trace[-1]['noise'] if trace and trace[-1]['noise'] is not None else None,
        'wall_seconds': time.perf_counter() - t0,
        'trace': trace,
    }
