import glfw
import moderngl
from camera import Camera
from particle_system import ParticleSystem

class App:
    def __init__(self, width=512, height=512, title="Fluoddity-Core"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)


        self.camera = Camera(self.ctx)
        self.system = ParticleSystem(self.ctx)

        glfw.set_key_callback(self.window, self._on_key)

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:
                self.system.reset()

    def poll_events(self):
        glfw.poll_events()

    def run(self):
        while not glfw.window_should_close(self.window):
            for i in range(3):
                self.system.advance()
            self.ctx.screen.use()
            self.ctx.clear(0., 0., 0., 1.0)
            self.camera.render_texture(self.system.brush_texture, self.ctx.screen)

            self.poll_events()

            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()
