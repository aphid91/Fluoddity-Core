import moderngl
import numpy as np
from utilities.gl_helpers import read_shader, tryset


class Camera:
    def __init__(self, ctx):
        self.ctx = ctx
        self.program = None
        self.vao = None
        self.reload()

    def reload(self):
        """Reload shaders from disk. Safe to call mid-execution."""
        try:
            vert_source = read_shader('shaders/camera.vert')
            frag_source = read_shader('shaders/camera.frag')

            new_program = self.ctx.program(
                vertex_shader=vert_source,
                fragment_shader=frag_source
            )

            # Only replace program if compilation succeeded
            self.program = new_program

            # Create fullscreen quad VAO if it doesn't exist
            if self.vao is None:
                vertices = self.ctx.buffer(np.array([
                    -1, -1,
                     1, -1,
                     1,  1,
                    -1, -1,
                     1,  1,
                    -1,  1,
                ], dtype=np.float32).tobytes())
                self.vao = self.ctx.vertex_array(
                    self.program,
                    [(vertices, '2f', 'in_position')]
                )

            print("Camera shaders reloaded successfully")

        except Exception as e:
            print(f"Failed to reload camera shaders: {e}")

    def render_texture(self, texture, framebuffer):
        """Render a texture to a framebuffer using a fullscreen quad."""
        if self.program is None or self.vao is None:
            return

        framebuffer.use()
        tryset(self.program, 'tex', 0)
        texture.use(location=0)
        self.vao.render(moderngl.TRIANGLES)
