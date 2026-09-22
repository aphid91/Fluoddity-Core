"""Headless harness around Fluoddity-Core's ParticleSystem.

Design rule: the physics is never reimplemented here. The rig drives the
unmodified ParticleSystem, ParticleSystem's own shaders, and reads state out.
The only extra GPU work is an additive accumulator pass (shaders/accumulate.frag)
that never writes back into the simulation.

Units. The simulation works in world coordinates spanning [-1, 1]. Everything
the rig reports is converted to *pixels* and *steps*:

    pixels_per_world_unit = canvas_dim / 2

so a world displacement of 0.01 on a 512-wide canvas is 2.56 px. Sensor
distance is `0.01 * sensor_distance / SQRT_WORLD_SIZE` world units, which at
defaults is ~2.56 px -- the pattern scale is only a couple of pixels, which is
why coarsening the grid is not an option (see spec section 2.2).
"""

import os
import subprocess
import time

import moderngl
import numpy as np

import particle_system as ps_mod
from gl_utils import read_shader
from particle_system import CANVAS_DIM, ENTITY_COUNT, SIZE_OF_ENTITY_STRUCT, ParticleSystem

# Entity struct is 6 floats: pos.xy, vel.xy, size, padding
FLOATS_PER_ENTITY = SIZE_OF_ENTITY_STRUCT // 4
RIG_SHADER_DIR = os.path.join(os.path.dirname(__file__), 'shaders')


def create_context():
    """A standalone GL 4.3+ context with no window attached."""
    return moderngl.create_standalone_context(require=430, backend='egl')


def gpu_info(ctx):
    """Identify the machine, so results measured on software rendering are never
    mistaken for results measured on a real GPU."""
    return {
        'GL_RENDERER': ctx.info.get('GL_RENDERER'),
        'GL_VERSION': ctx.info.get('GL_VERSION'),
        'GL_VENDOR': ctx.info.get('GL_VENDOR'),
        # llvmpipe/softpipe mean CPU rasterisation: throughput numbers from such
        # a run are not comparable to GPU numbers and must not anchor cost claims.
        'software_rendered': any(
            s in str(ctx.info.get('GL_RENDERER', '')).lower()
            for s in ('llvmpipe', 'softpipe', 'swrast')
        ),
    }


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


class Rig:
    """Drives one ParticleSystem and exposes the measurement hooks the rig needs."""

    def __init__(self, config_path, ctx=None):
        self.ctx = ctx if ctx is not None else create_context()
        self.system = ParticleSystem(self.ctx, config_path=config_path)
        self.config_path = config_path
        self.canvas_dim = self.system.canvas_size[0]
        self.px_per_world = self.canvas_dim / 2.0
        self._init_accumulator()

    # ---- derived config quantities ------------------------------------------

    @property
    def config(self):
        return self.system.config

    @property
    def persistence(self):
        """The value the shader actually uses, after canvas.frag's clamp."""
        return float(np.clip(self.config['trail_persistence'], 0.0, 0.999))

    @property
    def memory_time(self):
        """Steps a canvas pixel remembers a deposit for: 1/(1-p)."""
        return 1.0 / (1.0 - self.persistence)

    @property
    def particles_per_pixel(self):
        return ENTITY_COUNT / float(self.canvas_dim ** 2)

    # ---- stepping ------------------------------------------------------------

    def step(self, n=1):
        """n full simulation steps, exactly as main.py runs them."""
        for _ in range(n):
            self.system.advance()

    def step_frozen(self, n=1, accumulate=True):
        """n steps with the canvas held fixed.

        This is E2's core manipulation: particles keep moving and depositing,
        but the field they sense never changes. advance() is unrolled here
        rather than reused because the whole point is to omit update_canvas().
        frame_count still advances, because it seeds the hazard-reset RNG.
        """
        for _ in range(n):
            self.ctx.memory_barrier()
            self.system.create_brush()
            if accumulate:
                self.accumulate_brush()
            self.ctx.memory_barrier()
            self.system.update_entities()
            self.system.frame_count += 1

    def step_canvas_only(self, n=1):
        """n canvas updates with no particle motion and no brush rebuild.

        Used by E0.3: the brush texture is left holding whatever was written
        into it, so the canvas sees a constant source term and its evolution
        can be checked against the closed-form propagator.
        """
        for _ in range(n):
            self.system.update_canvas()
            self.system.frame_count += 1

    # ---- brush accumulation --------------------------------------------------

    def _init_accumulator(self):
        size = self.system.canvas_size
        self.accum_texture = self.ctx.texture(size, 4, dtype='f4')
        self.accum_fbo = self.ctx.framebuffer(color_attachments=[self.accum_texture])
        self.accum_steps = 0
        self.accum_program = self.ctx.program(
            vertex_shader=read_shader('shaders/fullscreen_quad.vert'),
            fragment_shader=read_shader(os.path.join(RIG_SHADER_DIR, 'accumulate.frag')),
        )
        verts = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self._accum_vbo = self.ctx.buffer(verts.tobytes())
        self.accum_vao = self.ctx.vertex_array(
            self.accum_program, [(self._accum_vbo, '2f', 'in_position')]
        )
        self.reset_accumulator()

    def reset_accumulator(self):
        self.accum_fbo.use()
        self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.accum_steps = 0

    def accumulate_brush(self):
        """Add the current brush texture into the accumulator (additive blend)."""
        self.accum_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.accum_program['src'] = 0
        self.system.brush_texture.use(location=0)
        self.accum_vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)
        self.accum_steps += 1

    def read_accumulator(self, mean=True):
        """(H, W, 4) accumulated brush. mean=True divides by the step count, giving
        the time-averaged deposit B-bar."""
        a = self._read_texture(self.accum_texture)
        if mean and self.accum_steps:
            a = a / float(self.accum_steps)
        return a

    # ---- state access --------------------------------------------------------

    def _read_texture(self, tex):
        return np.frombuffer(tex.read(), dtype=np.float32).reshape(
            self.canvas_dim, self.canvas_dim, 4
        )

    def read_canvas(self):
        return self._read_texture(self.system.canvas_texture)

    def read_brush(self):
        return self._read_texture(self.system.brush_texture)

    def write_canvas(self, data):
        self.system.canvas_texture.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())

    def write_brush(self, data):
        self.system.brush_texture.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())

    def read_entities(self):
        """All entities as (N, 6): pos.xy, vel.xy, size, padding."""
        return np.frombuffer(self.system.entity_buffer.read(), dtype=np.float32).reshape(
            -1, FLOATS_PER_ENTITY
        )

    def tracked_blocks(self, n_tracked=4096, n_blocks=None, seed=0):
        """Contiguous index blocks, stratified across cohorts.

        Contiguous rather than scattered because the entity buffer is read with
        offset+size: 4096 scattered indices would be 4096 separate reads per
        step, while the same coverage in contiguous blocks is a handful. Cohorts
        are contiguous index ranges, so one block per cohort keeps the sample
        stratified by cohort as well.
        """
        cohorts = max(1, int(self.config['cohorts']))
        n_blocks = n_blocks if n_blocks is not None else min(cohorts, 64)
        per_block = max(1, n_tracked // n_blocks)
        rng = np.random.default_rng(seed)
        span = ENTITY_COUNT // n_blocks
        blocks = []
        for b in range(n_blocks):
            lo = b * span
            hi = min(ENTITY_COUNT, lo + span) - per_block
            start = int(rng.integers(lo, max(lo + 1, hi))) if hi > lo else lo
            blocks.append((start, min(per_block, ENTITY_COUNT - start)))
        return blocks

    def read_tracked(self, blocks):
        """(sum(counts), 6) entity rows for the given (start, count) blocks."""
        out = []
        for start, count in blocks:
            raw = self.system.entity_buffer.read(
                size=count * SIZE_OF_ENTITY_STRUCT, offset=start * SIZE_OF_ENTITY_STRUCT
            )
            out.append(np.frombuffer(raw, dtype=np.float32).reshape(count, FLOATS_PER_ENTITY))
        return np.concatenate(out, axis=0)

    # ---- snapshot / restore --------------------------------------------------

    def snapshot(self):
        """Full simulation state. frame_count is included because it seeds the
        hazard-reset RNG -- restoring without it would silently change which
        particles get reset."""
        return {
            'entities': self.system.entity_buffer.read(),
            'canvas': self.system.canvas_texture.read(),
            'frame_count': self.system.frame_count,
            'config': dict(self.system.config),
        }

    def restore(self, snap):
        self.system.entity_buffer.write(snap['entities'])
        self.system.canvas_texture.write(snap['canvas'])
        self.system.frame_count = snap['frame_count']
        self.system.config = dict(snap['config'])

    # ---- misc ----------------------------------------------------------------

    def reset(self):
        """Back to frame 0. canvas.frag wipes the canvas to (0,0,0,1) on the next
        update, and entity_update.glsl re-places every particle."""
        self.system.reset()

    def benchmark(self, steps=50, warmup=5):
        """Steps per second with no readbacks, so the number reflects the sim and
        not the measurement apparatus."""
        for _ in range(warmup):
            self.system.advance()
        self.ctx.finish()
        t0 = time.perf_counter()
        for _ in range(steps):
            self.system.advance()
        self.ctx.finish()
        return steps / (time.perf_counter() - t0)
