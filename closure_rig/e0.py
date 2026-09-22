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
def blend_check(rig, n_probe=64, seed=0):
    """Splat particles with known velocity and see what the texture holds.

    brush.frag writes vec4(vel, 0.01, 1.0) * kernel, and the blend is
    (SRC_ALPHA, ONE) with src.a = kernel. So the framebuffer should accumulate
    vel*kernel^2 in .xy and kernel^2 in .w -- i.e. the stored deposit is
    kernel-SQUARED weighted, and .w is a density proxy only up to how much
    kernel^2 a particle happens to contribute from its sub-pixel position.

    That last caveat matters: the splat quad is ~1.5 px across, so where a
    particle sits inside its pixel changes how much of the Gaussian is sampled.
    This measures that spread, because it sets the noise floor on any density
    estimate built from brush.w.
    """
    N = rig.canvas_dim
    rng = np.random.default_rng(seed)
    vel = np.array([0.004, -0.002], dtype=np.float32)

    # --- single isolated particle: exact deposit relationship
    ents = np.zeros((ENTITY_COUNT, FLOATS_PER_ENTITY), dtype=np.float32)
    ents[0] = (0.0, 0.0, vel[0], vel[1], DEFAULT_SIZE, 0.0)
    rig.system.entity_buffer.write(ents.tobytes())
    rig.system.frame_count = 1            # brush.frag discards when frame_count == 0
    rig.system.create_brush()
    rig.ctx.finish()
    b = rig.read_brush()
    s = b.reshape(-1, 4).sum(axis=0)
    single = {
        'sum_xy': [float(s[0]), float(s[1])],
        'sum_z': float(s[2]),
        'sum_w': float(s[3]),
        'recovered_vel': [float(s[0] / s[3]), float(s[1] / s[3])],
        'true_vel': [float(vel[0]), float(vel[1])],
        'vel_recovered_exactly': bool(np.allclose(s[:2] / s[3], vel, rtol=1e-4)),
        'z_over_w': float(s[2] / s[3]),   # should be the 0.01 literal in brush.frag
        'footprint_pixels': int((np.abs(b[..., 3]) > 0).sum()),
    }

    # --- many well-separated particles at random sub-pixel offsets, one splat.
    # Spacing >> kernel width, so each particle's deposit can be summed locally.
    spacing = 32
    grid = np.arange(spacing // 2, N, spacing)
    cells = [(x, y) for y in grid for x in grid][:n_probe]
    ents = np.zeros((ENTITY_COUNT, FLOATS_PER_ENTITY), dtype=np.float32)
    for i, (px, py) in enumerate(cells):
        jx, jy = rng.random(2)            # sub-pixel offset within the cell
        wx = ((px + jx) / N) * 2.0 - 1.0
        wy = ((py + jy) / N) * 2.0 - 1.0
        ents[i] = (wx, wy, vel[0], vel[1], DEFAULT_SIZE, 0.0)
    rig.system.entity_buffer.write(ents.tobytes())
    rig.system.frame_count = 1
    rig.system.create_brush()
    rig.ctx.finish()
    b = rig.read_brush()
    r = 8
    masses = []
    for px, py in cells:
        y0, y1 = max(0, py - r), min(N, py + r + 1)
        x0, x1 = max(0, px - r), min(N, px + r + 1)
        masses.append(float(b[y0:y1, x0:x1, 3].sum()))
    masses = np.array(masses)
    total_w = float(b[..., 3].sum())
    return {
        'single_particle': single,
        'subpixel': {
            'n_probes': len(masses),
            'mean_kernel_sq': float(masses.mean()),
            'std_kernel_sq': float(masses.std()),
            'cv_percent': float(100.0 * masses.std() / masses.mean()),
            'min_over_max': float(masses.min() / masses.max()),
            # If local windows recover the global total, the splats really are
            # independent and additive blending is doing what we assume.
            'local_sums_match_global': bool(
                np.isclose(masses.sum(), total_w, rtol=1e-3)
            ),
        },
        'interpretation': (
            'brush.xy = sum(vel * k^2); brush.w = sum(k^2); brush.z = sum(0.01 * k^2). '
            'Mean deposited velocity m = brush.xy / brush.w is exact. '
            'brush.w is proportional to particle count only up to the sub-pixel '
            'spread reported above.'
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
def warmup(rig, cap=30000, probe_every=250, tol=0.02, patience=4, n_bands=16):
    """Run from reset until the canvas's radial power spectrum stops moving.

    The statistic is deliberately translation-invariant: we care that the
    pattern has reached its characteristic scales, not where the filaments are.

    Returns the step count at which it settled, or the cap with settled=False.
    A config that never settles is reported as such rather than having the
    threshold tuned until it passes.
    """
    rig.reset()
    rig.step(1)                            # step 1 wipes the canvas and places particles

    trace, prev, stable, settled_at = [], None, 0, None
    steps = 0
    t0 = time.perf_counter()
    while steps < cap:
        rig.step(probe_every)
        steps += probe_every
        spec, _ = prop.radial_power_spectrum(rig.read_canvas()[..., :2], n_bands=n_bands)
        total = spec.sum()
        if prev is not None:
            # L1 distance between successive spectra, normalised by total power:
            # a scale-free "how much did the pattern change" number.
            d = float(np.abs(spec - prev).sum() / max(total, 1e-30))
            trace.append({'step': steps, 'delta': d, 'total_power': float(total)})
            stable = stable + 1 if d < tol else 0
            if stable >= patience and settled_at is None:
                settled_at = steps - (patience - 1) * probe_every
                break
        else:
            trace.append({'step': steps, 'delta': None, 'total_power': float(total)})
        prev = spec

    return {
        'settled': settled_at is not None,
        'warmup_steps': settled_at if settled_at is not None else steps,
        'steps_run': steps,
        'cap': cap,
        'probe_every': probe_every,
        'tol': tol,
        'patience': patience,
        'memory_time_steps': rig.memory_time,
        'warmup_in_memory_times': (settled_at or steps) / rig.memory_time,
        'wall_seconds': time.perf_counter() - t0,
        'trace': trace,
    }
