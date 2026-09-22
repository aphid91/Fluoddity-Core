"""Closed-form propagator for the canvas update.

canvas.frag does, per pixel and per step:

    C_{t+1} = p * Blur(C_t) + (1 - p) * vec4(Brush_t.xy, 0, 1)

with p = clamp(trail_persistence, 0, 0.999) and Blur the 5-point stencil

    Blur(C) = (K*c + n + s + e + w) / (4 + K),
    K = 4 / (5^(td^2) - 1),   td = clamp(trail_diffusion, 0.001, 1).

IMPORTANT: only channels .xy obey this as a linear system driven by the brush.
canvas.frag discards Brush.z and Brush.w and substitutes the constants 0 and 1,
so C.z decays to zero and C.w relaxes to one regardless of what the particles
deposit. Density therefore cannot be recovered from the canvas; it has to come
from a brush accumulator. Everything in this module operates on .xy only.

Because Blur is a shift-invariant convolution on a periodic domain, it is
diagonal in Fourier space, so k steps cost one FFT rather than k shader passes.
That is what would let a reduced model take large timesteps on the canvas.
"""

import numpy as np


def diffusion_constant(trail_diffusion):
    """K, exactly as canvas.frag computes it (clamp, then square, then rescale)."""
    td = float(np.clip(trail_diffusion, 0.001, 1.0))
    return 4.0 / (5.0 ** (td * td) - 1.0)


def persistence(trail_persistence):
    return float(np.clip(trail_persistence, 0.0, 0.999))


def blur_symbol(shape, trail_diffusion):
    """Fourier symbol beta(kx, ky) of the 5-point stencil, shaped like fft2 output.

    Array axis 0 is y and axis 1 is x, matching how the canvas is read back.
    """
    h, w = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]
    K = diffusion_constant(trail_diffusion)
    return (K + 2.0 * np.cos(kx) + 2.0 * np.cos(ky)) / (4.0 + K)


def propagate(canvas_xy, brush_xy, trail_persistence, trail_diffusion, k):
    """Advance the canvas k steps under a brush held constant over the window.

    Per Fourier mode, with a = p*beta:

        C_k = a^k * C_0 + (1 - p) * (1 - a^k) / (1 - a) * B

    The geometric sum is exact, so k = 1000 costs the same as k = 1.
    """
    p = persistence(trail_persistence)
    beta = blur_symbol(canvas_xy.shape[:2], trail_diffusion)
    a = p * beta
    ak = a ** k
    # |a| <= p < 1 always, so (1 - a) never vanishes and no regularisation is needed.
    gain = (1.0 - p) * (1.0 - ak) / (1.0 - a)
    out = np.empty_like(canvas_xy, dtype=np.float64)
    for c in range(2):
        C = np.fft.fft2(canvas_xy[..., c])
        B = np.fft.fft2(brush_xy[..., c])
        out[..., c] = np.real(np.fft.ifft2(ak * C + gain * B))
    return out


def steady_state(brush_xy, trail_persistence, trail_diffusion):
    """The canvas a constant deposit would settle to: (1-p)(I - p*Blur)^{-1} B.

    At the k=0 mode beta is exactly 1, so the factor is (1-p)/(1-p) = 1 and the
    solve is well conditioned everywhere; no mode needs special handling.
    """
    p = persistence(trail_persistence)
    beta = blur_symbol(brush_xy.shape[:2], trail_diffusion)
    gain = (1.0 - p) / (1.0 - p * beta)
    out = np.empty_like(brush_xy, dtype=np.float64)
    for c in range(2):
        B = np.fft.fft2(brush_xy[..., c])
        out[..., c] = np.real(np.fft.ifft2(gain * B))
    return out


def radial_power_spectrum(field_xy, n_bands=16):
    """Power in |k| bands, summed over the two vector components.

    Used as the warm-up statistic: it is insensitive to where structures are and
    sensitive to how big they are, which is what "the pattern has stopped
    changing" should mean.
    """
    h, w = field_xy.shape[:2]
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    power = sum(np.abs(np.fft.fft2(field_xy[..., c])) ** 2 for c in range(2))
    edges = np.linspace(0, kr.max() + 1e-12, n_bands + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bands - 1)
    sums = np.bincount(idx, weights=power.ravel(), minlength=n_bands)
    counts = np.bincount(idx, minlength=n_bands)
    return sums / np.maximum(counts, 1), edges
