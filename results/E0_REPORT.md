# E0: sanity and calibration

Configs measured: **33**. Renderer: `llvmpipe (LLVM 20.1.2, 256 bits)`, GL `4.5 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.2`.

> **Software-rendered run.** Throughput below is CPU rasterisation and must not be used as the cost baseline for any speedup claim. Re-run E0.1 on a GPU for that number. Every other E0 result is renderer-independent.

## E0.2 Blend check

- `brush.xy = sum(vel * k^2)`, `brush.w = sum(k^2)`, `brush.z = sum(0.01 * k^2)`.

- Velocity recovered from `brush.xy / brush.w`: **exact** (max deviation 4.7e-10 across all offsets).

- Splat quad is 1.54 px wide; the Gaussian is sigma = 0.25 px, point-sampled at 1-4 pixel centres.

- Deposit per particle varies **728x** with sub-pixel position (CV 124%), so `brush.w` carries 39% of the information of a true count.

- This test is config-independent (no config uniform reaches brush.frag), so it is reported once rather than per config.


## E0.3 Propagator

Shader vs closed-form Fourier update, channels `.xy`, k up to 1000.


- Worst config: **Wreath**, relative L2 error 3.16e-06

- Best config: **Critters**, 6.92e-08


## E0.4 Warm-up

| config | memory time | warm-up steps | in memory times | evals | verdict |
|---|---|---|---|---|---|
| Critters | 1.8 | 2000 | 1108 | 11 | settled |
| Bubbles | 17.5 | 2400 | 137 | 15 | settled |
| SideWinder | 52.6 | 2500 | 47 | 16 | settled |
| Pop | 200.0 | 1000 | 5 | 1 | **floor-limited** |

Floor-limited means the drift test passed on its first evaluation, so the figure is the earliest step the window can credit rather than a measured settling time.


## Per-config table

| config | particles/px | memory time (steps) | steps/sec | propagator max rel err | warm-up |
|---|---|---|---|---|---|
| 9LeafClovers | 0.57 | 17.5 | 8.5 | 9.0e-07 | - |
| Adrift | 0.57 | 52.6 | 8.6 | 2.8e-06 | - |
| Angles | 0.57 | 17.5 | 7.5 | 9.0e-07 | - |
| Batarangs | 0.57 | 55.6 | 10.3 | 3.1e-06 | - |
| Bramble | 0.57 | 17.5 | 10.5 | 9.0e-07 | - |
| Bubbles | 0.57 | 17.5 | 9.6 | 9.0e-07 | 2400 |
| Butterflies | 0.57 | 16.1 | 9.3 | 7.7e-07 | - |
| Cilia | 0.57 | 17.5 | 9.7 | 9.0e-07 | - |
| Circuits | 0.57 | 88.3 | 7.9 | 2.5e-06 | - |
| Critters | 0.57 | 1.8 | 11.1 | 6.9e-08 | 2000 |
| Curls | 0.57 | 17.5 | 10.6 | 9.0e-07 | - |
| Dandelions | 0.57 | 52.6 | 9.3 | 2.8e-06 | - |
| Ferns | 0.57 | 16.1 | 8.6 | 7.7e-07 | - |
| Growth | 0.57 | 100.0 | 8.6 | 6.0e-07 | - |
| HungryHungryHippos | 0.57 | 55.6 | 9.5 | 3.1e-06 | - |
| LavaLamp | 0.57 | 16.1 | 7.7 | 7.7e-07 | - |
| Meandering | 0.57 | 52.6 | 10.3 | 2.8e-06 | - |
| Nettle | 0.57 | 200.0 | 9.1 | 9.9e-07 | - |
| Pop | 0.57 | 200.0 | 9.3 | 9.9e-07 | 1000 (floor-limited) |
| RingOfFire | 0.57 | 8.1 | 9.3 | 7.5e-07 | - |
| RippleTree | 0.57 | 16.1 | 10.9 | 7.7e-07 | - |
| Salt | 0.57 | 52.6 | 9.3 | 2.8e-06 | - |
| Searching | 0.57 | 55.6 | 9.8 | 3.1e-06 | - |
| SideWinder | 0.57 | 52.6 | 10.4 | 2.8e-06 | 2500 |
| Streamers | 0.57 | 52.6 | 10.9 | 2.8e-06 | - |
| TV | 0.57 | 16.1 | 7.9 | 7.7e-07 | - |
| Tadpole | 0.57 | 111.1 | 10.3 | 6.3e-07 | - |
| Vacuoles | 0.57 | 17.5 | 8.7 | 9.0e-07 | - |
| Veins | 0.57 | 55.6 | 9.2 | 3.1e-06 | - |
| Web | 0.57 | 52.6 | 10.9 | 2.8e-06 | - |
| Wreath | 0.57 | 62.5 | 9.3 | 3.2e-06 | - |
| Zipper | 0.57 | 83.3 | 10.1 | 7.7e-07 | - |
| _Default | 0.57 | 16.1 | 9.9 | 7.7e-07 | - |
