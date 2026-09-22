# closure_rig

A headless measurement rig for Fluoddity-Core. It **measures** the existing
simulation and builds no approximate model.

The question it exists to answer: can the particle system be replaced by a cheap
model of its slow, local, statistical behaviour instead of simulating every
particle at 2+ kHz? That only works if (a) fast and slow timescales are actually
separated, (b) the time-averaged deposit converges while the canvas is held
fixed, and (c) that averaged deposit is predictable from a small local patch of
canvas. The rig measures all three before anything gets built.

## Install

```
pip install -r closure_rig/requirements.txt
```

Needs a GL 4.3+ context. On a headless box that means EGL plus a driver:
`apt-get install libegl1 libegl-mesa0 libgl1-mesa-dri`. Mesa's llvmpipe works
but is CPU rasterisation — roughly 200x slower than a GPU, and the rig marks
such runs `software_rendered` so their throughput never anchors a speedup claim.

## Run

```
python -m closure_rig.run --config physics_configs/Pop.json --out results/
python -m closure_rig.run_all --out results/ [--fluoddity-path ../Fluoddity] [--quick]
python -m closure_rig.run_all --out results/ --audit-only     # just the audit
```

Useful flags: `--quick` (short runs, smoke testing), `--skip-warmup` (E0.4 is
the only expensive part of E0), `--warmup-cap N`, `--configs A B C`.

## Layout

| file | what it is |
|---|---|
| `harness.py` | drives the unmodified `ParticleSystem`; stepping, frozen-canvas stepping, brush accumulation, snapshot/restore, tracked readback |
| `propagator.py` | closed-form Fourier propagator for the canvas update, and the radial power spectrum used as the warm-up statistic |
| `configs.py` | config discovery across all three sources, de-duplication, Core-semantics audit |
| `e0.py` | E0.1 throughput, E0.2 blend check, E0.3 propagator test, E0.4 warm-up |
| `run.py` / `run_all.py` | single-config and batch CLIs |
| `shaders/accumulate.frag` | additive accumulation pass; **not** part of the physics |

## Notes for whoever picks this up

- **No physics shader is modified or copied.** The rig drives `shaders/*` as
  they ship. The spec's bit-identity test for instrumented shaders therefore
  does not apply yet; it will if E1-E3 ever need instrumentation.
- **Units** are pixels and steps everywhere in the output. World-to-pixel is
  `canvas_dim / 2`.
- **Only canvas channels `.xy` are physical.** `canvas.frag` discards the
  brush's `.z`/`.w` and substitutes constants, so `.z` decays to 0 and `.w` to 1
  no matter what the particles deposit. Density must come from a brush
  accumulator, never from the canvas.
- **`frame_count` is part of the state.** It seeds the hazard-reset RNG, so
  `snapshot()`/`restore()` carry it. It is also the reset signal: at 0,
  `canvas.frag` wipes the canvas and `entity_update.glsl` re-places every
  particle.
- **Tracked particles are read as contiguous blocks**, one per cohort, rather
  than as scattered indices. Cohorts are contiguous index ranges, so this stays
  stratified while costing a handful of reads per step instead of thousands.
