import glob
import os
import glfw
import moderngl
from camera import Camera
from particle_system import ParticleSystem

CONFIG_DIR = 'physics_configs'


def list_configs():
    """Every config in CONFIG_DIR, in the order the arrow keys cycle through them."""
    return sorted(os.path.normpath(p) for p in glob.glob(os.path.join(CONFIG_DIR, '*.json')))


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

        # Preset cycling starts wherever the loaded config sits in the list
        self.title = title
        self.configs = list_configs()
        current = os.path.normpath(self.system.config_path)
        self.config_index = self.configs.index(current) if current in self.configs else 0
        self.show_config(self.system.config_path)

        glfw.set_key_callback(self.window, self.on_key)

    def on_key(self, window, key, scancode, action, mods):
        """R restarts the simulation, left/right arrows step through CONFIG_DIR."""
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_R:
            self.system.reset()
        elif key == glfw.KEY_LEFT:
            self.cycle_config(-1)
        elif key == glfw.KEY_RIGHT:
            self.cycle_config(1)

    def cycle_config(self, step):
        """Load the config `step` places away, wrapping at either end of the list."""
        if not self.configs:
            print(f"No configs found in {CONFIG_DIR}/")
            return
        self.config_index = (self.config_index + step) % len(self.configs)
        path = self.configs[self.config_index]
        if self.system.set_config(path):
            self.show_config(path)

    def show_config(self, path):
        """Name the running config. There is no UI, so the title bar is where it goes."""
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Config: {name}")
        glfw.set_window_title(self.window, f"{self.title} - {name}")

    def run(self):
        while not glfw.window_should_close(self.window):
            #physics update hardcoded to 180hz
            for i in range(30):
                self.system.advance()
            
            self.ctx.screen.use()
            self.ctx.clear(0., 0., 0., 1.0)
            
            #display brush texture with camera.frag
            self.camera.render_texture(self.system.brush_texture, self.ctx.screen)

            glfw.poll_events()
            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()
