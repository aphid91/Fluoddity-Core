import math
import numpy as np
import moderngl
from utilities.gl_helpers import read_shader, tryset

ENTITY_COUNT = 150000
SIZE_OF_ENTITY_STRUCT = 48


class ParticleSystem:
    def __init__(self, ctx, canvas_size=(512, 512)):
        self.ctx = ctx
        self.canvas_size = canvas_size

        # Programs (initialized in reload)
        self.entity_update_program = None
        self.brush_splat_program = None
        self.canvas_update_program = None

        # Textures and framebuffers
        self.brush_texture = self.ctx.texture(canvas_size, 4, dtype='f4')
        self.brush_fbo = self.ctx.framebuffer(color_attachments=[self.brush_texture])

        self.canvas_texture = self.ctx.texture(canvas_size, 4, dtype='f4')
        self.canvas_fbo = self.ctx.framebuffer(color_attachments=[self.canvas_texture])

        # Double buffer for canvas update (read from one, write to other)
        self.canvas_texture_back = self.ctx.texture(canvas_size, 4, dtype='f4')
        self.canvas_fbo_back = self.ctx.framebuffer(color_attachments=[self.canvas_texture_back])

        # Entity buffer
        self.entity_buffer = self.ctx.buffer(reserve=ENTITY_COUNT * SIZE_OF_ENTITY_STRUCT)

        # Fullscreen quad for canvas update
        self.quad_vbo = None
        self.canvas_vao = None

        # Frame counter
        self.frame_count = 0

        # Initialize shaders
        self.reload()

        # Initialize entities with random positions and velocities
        self._init_entities()

    def _init_entities(self):
        """Initialize entity buffer with random data for testing."""
        # Entity struct: pos(2) + vel(2) + size(1) + cohort(1) + padding(2) + color(4) = 12 floats
        data = np.zeros((ENTITY_COUNT, 12), dtype=np.float32)

        # Random positions in [-1, 1]
        data[:, 0] = np.random.uniform(-1, 1, ENTITY_COUNT)  # pos.x
        data[:, 1] = np.random.uniform(-1, 1, ENTITY_COUNT)  # pos.y

        # Small random velocities
        data[:, 2] = np.random.uniform(-0.001, 0.001, ENTITY_COUNT)  # vel.x
        data[:, 3] = np.random.uniform(-0.001, 0.001, ENTITY_COUNT)  # vel.y

        # Size
        data[:, 4] = 0.005  # size

        # Cohort
        data[:, 5] = np.random.uniform(0, 1, ENTITY_COUNT)  # cohort

        # Padding (indices 6, 7) - leave as zeros

        # Color (RGBA)
        data[:, 8] = 1.0   # R
        data[:, 9] = 1.0   # G
        data[:, 10] = 1.0  # B
        data[:, 11] = 1.0  # A

        self.entity_buffer.write(data.tobytes())

    def reload(self):
        """Reload all shaders from disk. Safe to call mid-execution."""
        self._reload_entity_update()
        self._reload_brush_splat()
        self._reload_canvas_update()

    def _reload_entity_update(self):
        """Reload entity update compute shader."""
        try:
            source = read_shader('shaders/entity_update.glsl')
            new_program = self.ctx.compute_shader(source)
            self.entity_update_program = new_program
            print("Entity update shader reloaded successfully")
        except Exception as e:
            print(f"Failed to reload entity update shader: {e}")

    def _reload_brush_splat(self):
        """Reload brush splat shaders."""
        try:
            vert_source = read_shader('shaders/brush.vert')
            frag_source = read_shader('shaders/brush.frag')
            new_program = self.ctx.program(
                vertex_shader=vert_source,
                fragment_shader=frag_source
            )
            self.brush_splat_program = new_program
            print("Brush splat shaders reloaded successfully")
        except Exception as e:
            print(f"Failed to reload brush splat shaders: {e}")

    def _reload_canvas_update(self):
        """Reload canvas update shaders."""
        try:
            vert_source = read_shader('shaders/canvas.vert')
            frag_source = read_shader('shaders/canvas.frag')
            new_program = self.ctx.program(
                vertex_shader=vert_source,
                fragment_shader=frag_source
            )
            self.canvas_update_program = new_program

            # Create or recreate VAO with new program
            if self.quad_vbo is None:
                # Fullscreen quad vertices as floats
                vertices = np.array([
                    -1, -1,
                     1, -1,
                     1,  1,
                    -1, -1,
                     1,  1,
                    -1,  1,
                ], dtype=np.float32)
                self.quad_vbo = self.ctx.buffer(vertices.tobytes())

            self.canvas_vao = self.ctx.vertex_array(
                self.canvas_update_program,
                [(self.quad_vbo, '2f', 'in_position')]
            )
            print("Canvas update shaders reloaded successfully")
        except Exception as e:
            print(f"Failed to reload canvas update shaders: {e}")

    def advance(self):
        """Run one simulation step: update entities, create brush, update canvas."""
        self.ctx.memory_barrier()
        self.update_entities()
        self.ctx.memory_barrier()
        self.create_brush()
        self.ctx.memory_barrier()
        self.update_canvas()
        self.frame_count += 1

    def reset(self):
        """Reset simulation state."""
        self.frame_count = 0
        self._init_entities()

    def update_entities(self):
        """Dispatch compute shader to update entity positions."""
        if self.entity_update_program is None:
            return

        self.entity_buffer.bind_to_storage_buffer(0)

        tryset(self.entity_update_program, 'canvas_texture', 0)
        tryset(self.entity_update_program, 'frame_count', self.frame_count)
        self.canvas_texture.use(location=0)

        # Dispatch enough workgroups to cover all entities
        # local_size_x = 256, so we need ceil(ENTITY_COUNT / 256) workgroups
        workgroups = math.ceil(ENTITY_COUNT / 256)
        self.entity_update_program.run(workgroups, 1, 1)

        # Memory barrier to ensure writes are visible
        self.ctx.memory_barrier()

    def create_brush(self):
        """Splat all entities to brush texture as gaussian dots."""
        if self.brush_splat_program is None:
            return

        self.brush_fbo.use()
        self.brush_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Enable additive blending for overlapping particles
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        # Bind entity buffer as SSBO
        self.entity_buffer.bind_to_storage_buffer(0)

        # Set uniforms
        tryset(self.brush_splat_program, 'canvas_resolution',
               (float(self.canvas_size[0]), float(self.canvas_size[1])))
        tryset(self.brush_splat_program, 'frame_count', self.frame_count)

        # Instanced rendering: 4 vertices per quad, ENTITY_COUNT instances
        # No VAO needed - brush.vert generates vertices from gl_VertexID and gl_InstanceID
        vao = self.ctx.vertex_array(self.brush_splat_program, [])
        vao.render(moderngl.TRIANGLE_FAN, vertices=4, instances=ENTITY_COUNT)

        # Restore default blend mode
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def update_canvas(self):
        """Mix brush texture into canvas texture with trail persistence."""
        if self.canvas_update_program is None or self.canvas_vao is None:
            return

        # Render to back buffer, reading from front
        self.canvas_fbo_back.use()

        tryset(self.canvas_update_program, 'brush_texture', 0)
        tryset(self.canvas_update_program, 'canvas_texture', 1)
        tryset(self.canvas_update_program, 'frame_count', self.frame_count)

        self.brush_texture.use(location=0)
        self.canvas_texture.use(location=1)

        self.canvas_vao.render(moderngl.TRIANGLES)

        # Swap buffers
        self.canvas_texture, self.canvas_texture_back = self.canvas_texture_back, self.canvas_texture
        self.canvas_fbo, self.canvas_fbo_back = self.canvas_fbo_back, self.canvas_fbo
