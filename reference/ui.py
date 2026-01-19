import glfw
from imgui_bundle import imgui
from imgui_bundle.python_backends import glfw_backend
import time
import random
import numpy as np
import moderngl
from pathlib import Path
from dataclasses import dataclass
from state import UIState, SimState, CameraState, RecordingState
from services.config_saver import ConfigSaver, PhysicsConfig
from utilities.keybinding_management import KeybindingManager


@dataclass
class PhysicsDefaults:
    """Stores default physics values for reset functionality."""
    values: dict[str, float]
    source_filename: str | None  # None means program defaults



class UI:
    """Passive UI - renders widgets, exposes state, handles no logic."""

    def __init__(self, window, ctx: moderngl.Context, view_option_labels: list[str], multi_load_service=None):
        self.window = window
        self.ctx = ctx
        self.view_option_labels = view_option_labels
        self.multi_load_service = multi_load_service

        # Initialize keybinding manager
        self.keybindings = KeybindingManager()

        # Initialize ImGui
        imgui.create_context()
        self.imgui_renderer = glfw_backend.GlfwRenderer(window)

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking

        # Default font at normal size
        io.fonts.add_font_default()

        # Default font
        font_config = imgui.ImFontConfig()
        self.default_font = io.fonts.add_font_default(font_config)

        # Set up event callbacks
        self.setup_callbacks()

        # Create tooltip shader and texture
        self.tooltip_texture_size = 128
        self.setup_tooltip_shader()

        # UI-only state
        self.show_demo_window = False
        self.show_physics_settings_window = True  # Physics settings window (always visible, but can be hidden with sidebar)
        self.show_video_recording_window = False  # Video recording controls window
        self.show_sidebar = True  # Controls visibility of Physics Settings and Preferences windows

        # Rule history window state
        self.show_history_window = False
        self.history_window_labels: list[tuple[int, str, str]] = []  # [(jersey_num, digit1_rgb, digit2_rgb), ...]
        self.currently_previewing_index: int | None = None

        # Tooltip state - track which slider was last hovered
        self.last_hovered_slider = None
        self.last_hovered_description = ""
        self.physics_window_interaction = False  # Track if we're interacting with sliders
        self.tooltip_start_time = time.time()  # Track time for animations

        # File save/load state
        self.save_popup_open = False
        self.save_filename_buffer = ""
        self.configs_dir = Path("physics_configs")
        self.config_saver = ConfigSaver()

        # Load submenu preview state
        self.config_files: list[str] = []  # List of available config filenames (DEPRECATED: use config_files_by_category)
        self.config_files_by_category: dict[str, list[str]] = {}  # Config filenames organized by category (Core/Custom/Advanced)
        self.cached_configs: dict[str, PhysicsConfig] = {}  # Cached decoded configs
        self.load_submenu_was_open = False  # Track submenu open state
        self.cached_config: str | None = None  # JSON string of config when menu opened
        self.preview_rule_pushed: bool = False  # Whether we pushed a preview rule
        self.currently_previewing: str | None = None  # Currently hovered config
        self.currently_open_project: str = "_Default"  # Currently open project name
        # Track which load menu is open: None=neither, False=standard, True=watercolor
        self.load_menu_watercolor_mode: bool | None = None
        self._load_watercolor_override: bool | None = None  # Override for load operation

        # Delete confirmation state
        self.delete_confirm_filename: str | None = None

        # Overwrite confirmation state
        self.overwrite_confirm_filename: str | None = None

        # Menu auto-close state
        self.main_menu_bar_has_open_menu: bool = False  # Track if main menu bar has open menus
        self.physics_menu_bar_has_open_menu: bool = False  # Track if physics menu bar has open menus
        self.force_close_main_menus: bool = False  # Signal to close main menu bar menus
        self.force_close_physics_menus: bool = False  # Signal to close physics menu bar menus

        # Track last applied world size to detect changes
        self._last_applied_world_size: float = 1.0

        # State containers (Orchestrator reads these each frame)
        self.state = UIState(
            sim=SimState(),
            camera=CameraState(),
            recording=RecordingState()
        )

        # Initialize physics defaults with program defaults
        self.current_physics_defaults = PhysicsDefaults(
            values={
                'AXIAL_FORCE': self.state.sim.AXIAL_FORCE,
                'LATERAL_FORCE': self.state.sim.LATERAL_FORCE,
                'SENSOR_GAIN': self.state.sim.SENSOR_GAIN,
                'MUTATION_SCALE': self.state.sim.MUTATION_SCALE,
                'DRAG': self.state.sim.DRAG,
                'STRAFE_POWER': self.state.sim.STRAFE_POWER,
                'SENSOR_ANGLE': self.state.sim.SENSOR_ANGLE,
                'GLOBAL_FORCE_MULT': self.state.sim.GLOBAL_FORCE_MULT,
                'SENSOR_DISTANCE': self.state.sim.SENSOR_DISTANCE,
                'TRAIL_PERSISTENCE': self.state.sim.TRAIL_PERSISTENCE,
                'TRAIL_DIFFUSION': self.state.sim.TRAIL_DIFFUSION,
            },
            source_filename=None
        )

        # Input state (updated by callbacks)
        self._keys_pressed = set()
        self._mouse_pos = (0.0, 0.0)

        # One-shot flags (reset after get_state)
        self._left_click_pending = False
        self._right_click_pending = False
        self._any_left_click_pending = False  # Includes imgui clicks
        self._any_right_click_pending = False  # Includes imgui clicks
        self._scroll_delta = 0.0
        self._request_reload = False
        self._request_reset = False
        self._request_full_reset = False
        self._toggle_recording = False
        self._request_screenshot = False
        self._request_save_config = False
        self._request_load_config = False
        self._request_save_file = False
        self._request_load_file = False
        self._request_delete_file = False
        self._request_preview_config = False
        self._request_clear_preview = False
        self._request_world_size_change = False

        # Rule history window flags
        self._request_preview_history_rule = False
        self._request_clear_history_preview = False
        self._request_load_history_rule = False
        self._request_delete_history_rule = False
        self._history_preview_index = -1

        self._save_filename = ""
        self._load_filename = ""
        self._delete_filename = ""
        self._preview_filename = ""

        # Display info (received from Orchestrator)
        self._display_info = {
            'time': 0.0,
            'frame_count': 0,
            'tex_size': (1024, 1024),
            'recording_active': False,
        }

        # Resize debouncing
        self.pending_resize_time = None
        self.resize_debounce_delay = 0.15  # seconds

        # Timing for camera input
        self.last_update_time = time.time()

    def setup_callbacks(self):
        self.imgui_mouse_callback = glfw.set_mouse_button_callback(self.window, None)
        self.imgui_cursor_callback = glfw.set_cursor_pos_callback(self.window, None)
        self.imgui_scroll_callback = glfw.set_scroll_callback(self.window, None)
        self.imgui_key_callback = glfw.set_key_callback(self.window, None)
        self.imgui_char_callback = glfw.set_char_callback(self.window, None)

        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_char_callback(self.window, self.char_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

    def setup_tooltip_shader(self):
        """Create shader program and texture for tooltip graphics."""
        # Simple vertex shader for full-screen quad
        vert_shader = """
        #version 150
        in vec2 in_vert;
        out vec2 texcoord;
        void main() {
            texcoord = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        """

        # Load fragment shader
        with open('shaders/tooltip_graphic.frag', 'r') as f:
            frag_shader = f.read()

        # Create shader program
        self.tooltip_program = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader
        )

        # Create full-screen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype='f4')

        vbo = self.ctx.buffer(vertices.tobytes())
        self.tooltip_vao = self.ctx.vertex_array(
            self.tooltip_program,
            [(vbo, '2f', 'in_vert')]
        )

        # Create framebuffer and texture for rendering
        self.tooltip_texture = self.ctx.texture(
            size=(self.tooltip_texture_size, self.tooltip_texture_size),
            components=4
        )
        self.tooltip_fbo = self.ctx.framebuffer(color_attachments=[self.tooltip_texture])
        self.tooltip_texture_id = imgui.ImTextureRef(self.tooltip_texture.glo)

    def framebuffer_size_callback(self, window, width, height):
        # Debounce: just record the time, actual reload happens via request_reload flag
        self.pending_resize_time = time.time()

    def mouse_button_callback(self, window, button, action, mods):
        if self.imgui_mouse_callback:
            self.imgui_mouse_callback(window, button, action, mods)

        if action != glfw.PRESS:
            return

        # Track ALL clicks (including imgui) for sweep preview restore
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._any_left_click_pending = True
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._any_right_click_pending = True

        # Only track non-imgui clicks for normal interactions
        if imgui.get_io().want_capture_mouse:
            return

        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_click_pending = True
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_click_pending = True

    def cursor_pos_callback(self, window, xpos, ypos):
        if self.imgui_cursor_callback:
            self.imgui_cursor_callback(window, xpos, ypos)
        self._mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        if self.imgui_scroll_callback:
            self.imgui_scroll_callback(window, xoffset, yoffset)

        # Capture scroll for zoom-around-pointer (if imgui doesn't want it)
        if not imgui.get_io().want_capture_mouse:
            self._scroll_delta += yoffset

    def key_callback(self, window, key, scancode, action, mods):
        if self.imgui_key_callback:
            self.imgui_key_callback(window, key, scancode, action, mods)

        if imgui.get_io().want_capture_keyboard:
            return

        if action == glfw.PRESS:
            self._keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self._keys_pressed.discard(key)

        # One-shot key commands
        if action == glfw.PRESS:
            ctrl_pressed = mods & glfw.MOD_CONTROL
            shift_pressed = mods & glfw.MOD_SHIFT

            # Config save/load with Ctrl+C/Ctrl+V
            if ctrl_pressed and key == self.keybindings.get_key("copy_config_with_ctrl"):
                self._request_save_config = True
            elif ctrl_pressed and key == self.keybindings.get_key("paste_config_with_ctrl"):
                self._request_load_config = True
            elif key == self.keybindings.get_key("toggle_watercolor"):
                # Toggle watercolor mode
                self.state.sim.watercolor_mode = not self.state.sim.watercolor_mode
            elif key == self.keybindings.get_key("reload_shaders"):
                # Reload shaders
                self._request_reload = True
            elif key == self.keybindings.get_key("toggle_help"):
                # Show tutorial
                self.state.preferences.show_tutorial_window = not self.state.preferences.show_tutorial_window
            elif shift_pressed and key == self.keybindings.get_key("record_screen"):
                # Screenshot (Shift+P)
                self._request_screenshot = True
            elif key == self.keybindings.get_key("record_screen"):
                self._toggle_recording = True
            elif key == self.keybindings.get_key("toggle_pause"):
                self.state.sim.going = not self.state.sim.going
            elif key == self.keybindings.get_key("randomize_mutations"):
                self.state.sim.rule_seed = random.random()
            elif key == self.keybindings.get_key("toggle_mouse_mode"):
                # Toggle mouse mode between Select Particle and Draw Trail
                if self.state.preferences.mouse_mode == "Select Particle":
                    self.state.preferences.mouse_mode = "Draw Trail"
                else:
                    self.state.preferences.mouse_mode = "Select Particle"
            elif key == self.keybindings.get_key("toggle_parameter_sweep"):
                # Toggle parameter sweeps
                self.state.sim.parameter_sweeps_enabled = not self.state.sim.parameter_sweeps_enabled
                # If re-enabling sweeps while in preview mode, clear the preview flag
                if self.state.sim.parameter_sweeps_enabled and self.state.sim.sweep_preview_pending_restore:
                    self.state.sim.sweep_preview_pending_restore = False
            elif key == self.keybindings.get_key("randomize_rules"):
                # Full reset (one-shot, not hold)
                self._request_full_reset = True
            elif key == self.keybindings.get_key("toggle_sidebar"):
                # Toggle sidebar (Physics Settings and Preferences windows)
                self.show_sidebar = not self.show_sidebar
            elif key == self.keybindings.get_key("exit_keybinding"):
                glfw.set_window_should_close(window, True)
            #elif key == self.keybindings.get_key("toggle_tooltips"):
            #    self.show_demo_window = not self.show_demo_window

    def char_callback(self, window, char):
        if self.imgui_char_callback:
            self.imgui_char_callback(window, char)

    def get_state(self) -> UIState:
        """Return current UI state for Orchestrator to read.

        Returns snapshot and resets one-shot flags.
        """
        # Check for pending resize (debounced)
        if self.pending_resize_time is not None:
            if time.time() - self.pending_resize_time >= self.resize_debounce_delay:
                self._request_reload = True
                self.pending_resize_time = None

        # Check for R key hold (reset command - continuous)
        reset_key = self.keybindings.get_key("reset_keybinding")
        if reset_key and reset_key in self._keys_pressed:
            self._request_reset = True

        # Build state snapshot
        self.state.keys_pressed = self._keys_pressed.copy()
        self.state.mouse_pos = self._mouse_pos
        self.state.left_click_this_frame = self._left_click_pending
        self.state.right_click_this_frame = self._right_click_pending
        self.state.any_left_click_this_frame = self._any_left_click_pending
        self.state.any_right_click_this_frame = self._any_right_click_pending

        # Continuous mouse state (for draw trail mode) - respects imgui capture
        left_button_pressed = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.state.mouse_left_held = left_button_pressed and not imgui.get_io().want_capture_mouse
        self.state.scroll_delta = self._scroll_delta
        self.state.request_reload = self._request_reload
        self.state.request_reset = self._request_reset
        self.state.request_full_reset = self._request_full_reset
        self.state.toggle_recording = self._toggle_recording
        self.state.request_screenshot = self._request_screenshot
        self.state.request_save_config = self._request_save_config
        self.state.request_load_config = self._request_load_config
        self.state.request_save_file = self._request_save_file
        self.state.request_load_file = self._request_load_file
        self.state.request_delete_file = self._request_delete_file
        self.state.request_preview_config = self._request_preview_config
        self.state.request_clear_preview = self._request_clear_preview
        self.state.request_world_size_change = self._request_world_size_change
        self.state.save_filename = self._save_filename
        self.state.load_filename = self._load_filename
        self.state.delete_filename = self._delete_filename
        self.state.preview_filename = self._preview_filename
        self.state.load_watercolor_override = self._load_watercolor_override

        # Transfer history window flags
        self.state.request_preview_history_rule = self._request_preview_history_rule
        self.state.request_clear_history_preview = self._request_clear_history_preview
        self.state.request_load_history_rule = self._request_load_history_rule
        self.state.request_delete_history_rule = self._request_delete_history_rule
        self.state.history_preview_index = self._history_preview_index

        # Read clipboard content if load is requested
        if self._request_load_config:
            clipboard = glfw.get_clipboard_string(self.window)
            self.state.clipboard_text = clipboard if clipboard else ""
        else:
            self.state.clipboard_text = ""

        # Reset one-shot flags
        self._left_click_pending = False
        self._right_click_pending = False
        self._any_left_click_pending = False
        self._any_right_click_pending = False
        self._scroll_delta = 0.0
        self._request_reload = False
        self._request_reset = False
        self._request_full_reset = False
        self._toggle_recording = False
        self._request_screenshot = False
        self._request_save_config = False
        self._request_load_config = False
        self._request_save_file = False
        self._request_load_file = False
        self._request_delete_file = False
        self._request_preview_config = False
        self._request_clear_preview = False
        self._request_world_size_change = False
        self._save_filename = ""
        self._load_filename = ""
        self._delete_filename = ""
        self._preview_filename = ""
        self._load_watercolor_override = None

        # Reset history window flags
        self._request_preview_history_rule = False
        self._request_clear_history_preview = False
        self._request_load_history_rule = False
        self._request_delete_history_rule = False
        self._history_preview_index = -1

        return self.state

    def update_display_info(self, info: dict) -> None:
        """Receive read-only info for display (time, frame_count, etc.)."""
        self._display_info = info

    def set_clipboard(self, text: str) -> None:
        """Set clipboard content (used by orchestrator for config save)."""
        glfw.set_clipboard_string(self.window, text)

    def update_physics_defaults(self, filename: str) -> None:
        """Update current physics defaults from current sim state (called after file load/save)."""
        self.currently_open_project = filename
        self.current_physics_defaults = PhysicsDefaults(
            values={
                'AXIAL_FORCE': self.state.sim.AXIAL_FORCE,
                'LATERAL_FORCE': self.state.sim.LATERAL_FORCE,
                'SENSOR_GAIN': self.state.sim.SENSOR_GAIN,
                'MUTATION_SCALE': self.state.sim.MUTATION_SCALE,
                'DRAG': self.state.sim.DRAG,
                'STRAFE_POWER': self.state.sim.STRAFE_POWER,
                'SENSOR_ANGLE': self.state.sim.SENSOR_ANGLE,
                'GLOBAL_FORCE_MULT': self.state.sim.GLOBAL_FORCE_MULT,
                'SENSOR_DISTANCE': self.state.sim.SENSOR_DISTANCE,
                'TRAIL_PERSISTENCE': self.state.sim.TRAIL_PERSISTENCE,
                'TRAIL_DIFFUSION': self.state.sim.TRAIL_DIFFUSION,
            },
            source_filename=filename
        )

    def render(self):
        """Render ImGui widgets - modifies self.state based on widget interactions."""
        self.imgui_renderer.process_inputs()

        imgui.new_frame()

        # Create a full-window dockspace
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        imgui.set_next_window_viewport(viewport.id_)

        dockspace_window_flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_nav_focus
            | imgui.WindowFlags_.no_background
        )

        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))

        imgui.begin("DockSpace Window", None, dockspace_window_flags)
        imgui.pop_style_var(3)

        # Create the dockspace
        dockspace_id = imgui.get_id("MainDockSpace")
        imgui.dock_space(dockspace_id, imgui.ImVec2(0.0, 0.0), imgui.DockNodeFlags_.passthru_central_node)

        # Apply global color tinting based on mode
        recording_active = self._display_info.get('recording_active', False)
        sweeps_active = self.state.sim.parameter_sweeps_enabled
        color_push_count = 0

        if recording_active:
            # Red tint for video recording mode
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.3, 0.1, 0.1, 0.94))
            imgui.push_style_color(imgui.Col_.menu_bar_bg, imgui.ImVec4(0.35, 0.12, 0.12, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg, imgui.ImVec4(0.25, 0.08, 0.08, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg_active, imgui.ImVec4(0.4, 0.13, 0.13, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg_collapsed, imgui.ImVec4(0.25, 0.08, 0.08, 0.5))
            imgui.push_style_color(imgui.Col_.popup_bg, imgui.ImVec4(0.3, 0.1, 0.1, 0.94))
            imgui.push_style_color(imgui.Col_.header, imgui.ImVec4(0.4, 0.13, 0.13, 0.45))
            imgui.push_style_color(imgui.Col_.header_hovered, imgui.ImVec4(0.45, 0.15, 0.15, 0.8))
            imgui.push_style_color(imgui.Col_.header_active, imgui.ImVec4(0.5, 0.17, 0.17, 1.0))
            color_push_count = 9
        elif sweeps_active:
            # Yellow tint for parameter sweeps mode (30% less intense)
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.205, 0.19, 0.13, 0.94))
            imgui.push_style_color(imgui.Col_.menu_bar_bg, imgui.ImVec4(0.245, 0.231, 0.161, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg, imgui.ImVec4(0.17, 0.161, 0.119, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg_active, imgui.ImVec4(0.28, 0.266, 0.161, 1.0))
            imgui.push_style_color(imgui.Col_.title_bg_collapsed, imgui.ImVec4(0.17, 0.161, 0.119, 0.5))
            imgui.push_style_color(imgui.Col_.popup_bg, imgui.ImVec4(0.205, 0.19, 0.13, 0.94))
            imgui.push_style_color(imgui.Col_.header, imgui.ImVec4(0.28, 0.266, 0.161, 0.45))
            imgui.push_style_color(imgui.Col_.header_hovered, imgui.ImVec4(0.325, 0.3115, 0.175, 0.8))
            imgui.push_style_color(imgui.Col_.header_active, imgui.ImVec4(0.37, 0.357, 0.189, 1.0))
            color_push_count = 9

        # Main application menu bar
        self.render_main_menu_bar()

        # Render popup modals (Save, Overwrite, Delete) - always rendered regardless of sidebar
        self.render_popup_modals()

        # Render Physics Settings window if sidebar is visible
        if self.show_sidebar:
            self.render_physics_settings_window()

        # Render Preferences window if sidebar is visible AND preferences are enabled
        if self.show_sidebar and self.state.preferences.show_preferences_window:
            self.render_preferences_window()

        # Render Controls help window if visible
        if self.state.preferences.show_controls_window:
            self.render_controls_window()

        # Render Parameter Sweeps help window if visible
        if self.state.preferences.show_parameter_sweeps_window:
            self.render_parameter_sweeps_window()

        # Render Tutorial help window if visible
        if self.state.preferences.show_tutorial_window:
            self.render_tutorial_window()

        # Render Performance help window if visible
        if self.state.preferences.show_performance_window:
            self.render_performance_window()

        # Render Screen Recording window if visible
        if self.show_video_recording_window:
            self.render_video_recording_window()

        # Render history window if visible
        if self.show_history_window:
            self.render_history_window()

        if self.show_demo_window:
            imgui.show_demo_window()

        # Restore normal colors if we pushed any
        if color_push_count > 0:
            imgui.pop_style_color(color_push_count)

        # End dockspace window
        imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def render_main_menu_bar(self):
        """Render the main application menu bar at the top of the window."""
        load_submenu_open = False
        any_menu_open_this_frame = False

        if imgui.begin_main_menu_bar():
            # Track all open menu rectangles separately (not combined into one giant box)
            # We'll calculate distance as the minimum distance to any of these rectangles
            menu_rectangles = []

            # Start with the menu bar itself
            menu_bar_min = imgui.get_window_pos()
            menu_bar_size = imgui.get_window_size()
            menu_rectangles.append((menu_bar_min.x, menu_bar_min.y,
                                   menu_bar_min.x + menu_bar_size.x,
                                   menu_bar_min.y + menu_bar_size.y))
            if imgui.begin_menu("File", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                file_menu_min = imgui.get_window_pos()
                file_menu_size = imgui.get_window_size()
                menu_rectangles.append((file_menu_min.x, file_menu_min.y,
                                       file_menu_min.x + file_menu_size.x,
                                       file_menu_min.y + file_menu_size.y))

                if imgui.menu_item("New", "", False)[0]:
                    self._load_filename = "_Default"
                    self._request_load_file = True
                    self._load_watercolor_override = None
                self._delayed_tooltip("Start a fresh config. Loads from _Default")

                imgui.separator()

                if imgui.menu_item("Save...", "", False)[0]:
                    self.save_popup_open = True
                    # Default to last loaded filename
                    self.save_filename_buffer = self.currently_open_project
                self._delayed_tooltip("Save the current physics settings including particle rules.")

                # Load submenu with preview - locks to current watercolor mode
                # Right-click toggles watercolor mode
                if imgui.begin_menu("Load", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    load_submenu_open = True
                    # Add this submenu's bounding box to the list
                    load_menu_min = imgui.get_window_pos()
                    load_menu_size = imgui.get_window_size()
                    menu_rectangles.append((load_menu_min.x, load_menu_min.y,
                                           load_menu_min.x + load_menu_size.x,
                                           load_menu_min.y + load_menu_size.y))

                    # First frame submenu opens: cache current state and scan config files
                    if not self.load_submenu_was_open:
                        self._cache_all_configs()
                        # Cache current config as JSON string for restoration
                        self.cached_config = self.config_saver.save_to_string(
                            self.state.sim, self._display_info.get('current_rule'))
                        self.preview_rule_pushed = False
                        self.currently_previewing = None
                        # Lock to current watercolor mode when menu opens
                        self.load_menu_watercolor_mode = self.state.sim.watercolor_mode

                    # Use locked watercolor mode
                    current_menu_watercolor = self.load_menu_watercolor_mode

                    # Header showing right-click hint (compact two-line format)
                    mode_text = "Watercolor ON" if current_menu_watercolor else "Watercolor OFF"
                    imgui.text_disabled("Right-click toggles:")
                    imgui.text_disabled(f"({mode_text})")
                    imgui.separator()

                    # Check for right-click anywhere in the menu to toggle watercolor
                    if imgui.is_window_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.load_menu_watercolor_mode = not self.load_menu_watercolor_mode
                        current_menu_watercolor = self.load_menu_watercolor_mode
                        # Update any current preview with new watercolor mode
                        if self.currently_previewing and self.currently_previewing in self.cached_configs:
                            config = self.cached_configs[self.currently_previewing]
                            self.config_saver.apply_config(config, self.state.sim,
                                                          watercolor_override=current_menu_watercolor)
                        elif self.cached_config:
                            # Restore from cache with watercolor override
                            self.config_saver.load_from_string(
                                self.cached_config, self.state.sim,
                                watercolor_override=current_menu_watercolor)

                    # Lock watercolor mode to menu's mode
                    self.state.sim.watercolor_mode = current_menu_watercolor

                    hovered_this_frame = self._render_load_submenu_content(current_menu_watercolor)

                    # Handle preview on hover (works in both normal and multi-load modes)
                    if hovered_this_frame != self.currently_previewing:
                        # First, clear any existing preview
                        if self.currently_previewing:
                            self._request_clear_preview = True

                        if hovered_this_frame and hovered_this_frame in self.cached_configs:
                            # Apply preview config with watercolor override
                            config = self.cached_configs[hovered_this_frame]
                            self.config_saver.apply_config(config, self.state.sim,
                                                          watercolor_override=current_menu_watercolor)
                            self._request_preview_config = True
                            self._preview_filename = hovered_this_frame
                            self.currently_previewing = hovered_this_frame
                        elif hovered_this_frame is None and self.cached_config:
                            # Revert to cached state with watercolor override
                            self.config_saver.load_from_string(
                                self.cached_config, self.state.sim,
                                watercolor_override=current_menu_watercolor)
                            self.currently_previewing = None

                    imgui.end_menu()

                imgui.separator()

                # Preferences toggle
                if imgui.menu_item("Preferences", "", self.state.preferences.show_preferences_window)[0]:
                    self.state.preferences.show_preferences_window = not self.state.preferences.show_preferences_window

                imgui.end_menu()

            # Sidebar toggle button (shows/hides Physics Settings and Preferences)
            if imgui.menu_item("Show/Hide Sidebar (X)", "", self.show_sidebar)[0]:
                self.show_sidebar = not self.show_sidebar

            # Reset menu
            if imgui.begin_menu("Reset...", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                reset_menu_min = imgui.get_window_pos()
                reset_menu_size = imgui.get_window_size()
                menu_rectangles.append((reset_menu_min.x, reset_menu_min.y,
                                       reset_menu_min.x + reset_menu_size.x,
                                       reset_menu_min.y + reset_menu_size.y))

                # Revert to current project (reload the file)
                revert_label = f"Revert to '{self.currently_open_project}'"

                if imgui.menu_item(revert_label, "", False)[0]:
                    # Trigger file load equivalent to File->Load
                    self._load_filename = self.currently_open_project
                    self._request_load_file = True
                    self._load_watercolor_override = None  # Keep current watercolor mode
                self._delayed_tooltip(f"Equivalent to File -> Load {self.currently_open_project}")

                # Reset all slider ranges
                if imgui.menu_item("Reset all slider ranges to defaults", "", False)[0]:
                    # Clear all custom slider ranges, reverting to defaults
                    self.state.sim.slider_ranges.clear()

                # Reset all parameter sweeps
                if imgui.menu_item("Reset all parameter sweeps", "", False)[0]:
                    # Turn off all parameter sweeps
                    for param in list(self.state.sim.x_sweeps.keys()):
                        self.state.sim.x_sweeps[param] = 0.0
                        self.state.sim.y_sweeps[param] = 0.0
                        self.state.sim.cohort_sweeps[param] = 0.0
                self._delayed_tooltip("Set all parameter sweeps to 'off'.")

                # Reset all UI settings
                if imgui.menu_item("Reset all UI settings", "", False)[0]:
                    # Reset preferences to defaults (equivalent to deleting preferences.config)
                    from state.preferences_state import PreferencesState
                    self.state.preferences = PreferencesState()

                    
                self._delayed_tooltip("Restore all preferences and ui state to factory settings. \nEquivalent to deleting preferences.config, or running this\nprogram for the first time. Physics config saves are not affected.")

                imgui.end_menu()

            # Help menu
            if imgui.begin_menu("Help", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                help_menu_min = imgui.get_window_pos()
                help_menu_size = imgui.get_window_size()
                menu_rectangles.append((help_menu_min.x, help_menu_min.y,
                                       help_menu_min.x + help_menu_size.x,
                                       help_menu_min.y + help_menu_size.y))

                if imgui.menu_item("Controls", "", self.state.preferences.show_controls_window)[0]:
                    self.state.preferences.show_controls_window = not self.state.preferences.show_controls_window
                if imgui.menu_item("Parameter Sweeps", "", self.state.preferences.show_parameter_sweeps_window)[0]:
                    self.state.preferences.show_parameter_sweeps_window = not self.state.preferences.show_parameter_sweeps_window
                if imgui.menu_item("Tutorial", "", self.state.preferences.show_tutorial_window)[0]:
                    self.state.preferences.show_tutorial_window = not self.state.preferences.show_tutorial_window
                if imgui.menu_item("Performance", "", self.state.preferences.show_performance_window)[0]:
                    self.state.preferences.show_performance_window = not self.state.preferences.show_performance_window
                imgui.end_menu()

            # Extras menu
            if imgui.begin_menu("Extras", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                extras_menu_min = imgui.get_window_pos()
                extras_menu_size = imgui.get_window_size()
                menu_rectangles.append((extras_menu_min.x, extras_menu_min.y,
                                       extras_menu_min.x + extras_menu_size.x,
                                       extras_menu_min.y + extras_menu_size.y))

                # Multi Load toggle
                _, self.state.multi_load.multi_load_enabled = imgui.checkbox(
                    "Multi Load - EXPERIMENTAL",
                    self.state.multi_load.multi_load_enabled
                )
                self._delayed_tooltip("Load multiple files at once, so that particles\nfrom different saves can interact.")

                # Screen Recording Controls
                if imgui.menu_item("Screen Recording Controls", "", self.show_video_recording_window)[0]:
                    self.show_video_recording_window = not self.show_video_recording_window

                imgui.end_menu()

            # After all menus: check mouse distance from all menu rectangles
            # Find the minimum distance to any rectangle
            if self.main_menu_bar_has_open_menu and not self.save_popup_open:
                mouse_pos = imgui.get_mouse_pos()

                # Calculate minimum distance to any menu rectangle
                min_distance = float('inf')
                for min_x, min_y, max_x, max_y in menu_rectangles:
                    dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                    dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # If mouse is too far away from all rectangles, signal to close menus
                if min_distance > self.state.preferences.menu_close_threshold:
                    self.force_close_main_menus = True

            imgui.end_main_menu_bar()

        # Update menu tracking state
        self.main_menu_bar_has_open_menu = any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_main_menus and not any_menu_open_this_frame:
            self.force_close_main_menus = False

        # Handle submenu close without selection
        if self.load_submenu_was_open and not load_submenu_open:
            # Submenu just closed - restore cached state (no watercolor override)
            if self.cached_config:
                self.config_saver.load_from_string(self.cached_config, self.state.sim)
            if self.currently_previewing:
                self._request_clear_preview = True
            self.cached_config = None
            self.currently_previewing = None
            self.cached_configs = {}
            self.preview_rule_pushed = False
            self.load_menu_watercolor_mode = None  # Clear the watercolor lock

        self.load_submenu_was_open = load_submenu_open

    def render_preferences_window(self):
        """Render the Preferences window (closeable)."""
        recording_active = self._display_info.get('recording_active', False)

        # Apply red tint to window background when recording
        if recording_active:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

        # Use p_open to allow closing with X button
        expanded, self.state.preferences.show_preferences_window = imgui.begin("Preferences", True)

        if expanded:
            # === World Size section ===
            imgui.text("World Size")

            # Use input_float - only apply when user commits (Enter or focus loss)
            changed, new_value = imgui.input_float(
                "World Size",
                self.state.preferences.world_size,
                step=0.0,  # No step buttons
                step_fast=0.0,
                format="%.2f"
            )

            # Clamp to valid range
            if new_value < 0.02:
                new_value = 0.02
            elif new_value > 4.0:
                new_value = 4.0

            # Update the displayed value (clamping happens immediately)
            self.state.preferences.world_size = new_value

            # Only trigger world size change when user commits the edit
            if imgui.is_item_deactivated_after_edit():
                if abs(self.state.preferences.world_size - self._last_applied_world_size) > 0.001:
                    self._request_world_size_change = True

            self._delayed_tooltip("EXPENSIVE - Controls the size of the simulation world.\nAffects both entity count and canvas resolution to keep density ~fixed")

            imgui.separator()

            # === Physics Update Frequency section ===
            imgui.text("Physics Update Frequency")

            # Lock speedmult to motion_blur_samples when recording video
            if recording_active:
                locked_value = self.state.preferences.motion_blur_samples
                imgui.begin_disabled()
                imgui.slider_int(
                    label="Rate",
                    v=locked_value,
                    v_min=1,
                    v_max=6,
                    format=f"x{locked_value} ({locked_value * 60}hz) [locked]"
                )
                imgui.end_disabled()
            else:
                # Slider with custom format showing multiplier and hz
                current_hz = self.state.preferences.speedmult * 60
                _, self.state.preferences.speedmult = imgui.slider_int(
                    label="Rate",
                    v=self.state.preferences.speedmult,
                    v_min=1,
                    v_max=30,
                    format=f"x%d ({current_hz}hz)"
                )
            self._delayed_tooltip("EXPENSIVE- Multiple physics steps can be calculated each\nrender frame and blended together for faster physics.\nMotion blur can be costly for high frequencies,\ntry turning it off if things feel sluggish.")

            # Motion blur checkbox (lock during recording)
            if recording_active:
                imgui.begin_disabled()

            _, self.state.preferences.motion_blur = imgui.checkbox(
                "Motion Blur",
                self.state.preferences.motion_blur
            )
            self._delayed_tooltip("EXPENSIVE- Multiple physics steps can be calculated each\nrender frame and blended together for faster physics.\nMotion blur can be costly for high frequencies,\ntry turning it off if things feel sluggish.")

            if recording_active:
                imgui.end_disabled()

            # Blur Quality slider (only shown when motion blur is enabled)
            if self.state.preferences.motion_blur:
                imgui.indent(20)
                # Custom format for blur quality
                blur_val = self.state.preferences.blur_quality
                if blur_val == 1:
                    blur_format = "1 : Every Frame"
                else:
                    blur_format = f"{blur_val} : Every {blur_val} Frames"

                _, self.state.preferences.blur_quality = imgui.slider_int(
                    "Blur Quality",
                    self.state.preferences.blur_quality,
                    1, 20,
                    format=blur_format
                )
                self._delayed_tooltip("Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance")
                imgui.unindent(20)

            imgui.separator()

            # === Mouse Interaction section ===
            imgui.text("Mouse Interaction (Press 'T' to toggle)")

            # Mouse mode combo box (locked when multi-load enabled)
            if self.state.multi_load.multi_load_enabled:
                imgui.begin_disabled()
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Mouse Mode: Draw Trail (locked in Multi-Load)")
                imgui.end_disabled()
            else:
                mouse_modes = ["Select Particle", "Draw Trail"]
                current_mode_idx = mouse_modes.index(self.state.preferences.mouse_mode) if self.state.preferences.mouse_mode in mouse_modes else 0
                clicked, new_mode_idx = imgui.combo("Mouse Mode", current_mode_idx, mouse_modes)
                if clicked:
                    self.state.preferences.mouse_mode = mouse_modes[new_mode_idx]
                self._delayed_tooltip("In select Particle mode, clicking selects a particle rule to focus on.\nIn Draw trail mode, click and drag to leave trails on the canvas.\nSee Help->Controls for more")

            # Draw mode sliders (only show when in Draw Trail mode)
            if self.state.preferences.mouse_mode == "Draw Trail":
                imgui.indent(20)
                _, self.state.preferences.draw_size = imgui.slider_float(
                    "Draw Size",
                    self.state.preferences.draw_size,
                    0.01, 0.5,
                    format="%.3f"
                )
                _, self.state.preferences.draw_power = imgui.slider_float(
                    "Draw Power",
                    self.state.preferences.draw_power,
                    0.1, 5.0,
                    format="%.2f"
                )
                imgui.unindent(20)

            imgui.separator()

            # === View section ===
            imgui.text("View")

            # View dropdown
            changed, self.state.sim.current_view_option = imgui.combo(
                label="Current View",
                current_item=self.state.sim.current_view_option,
                items=self.view_option_labels + ['Camera (Particles rendered as dots)', 'Camera[Tiled] - EXPERIMENTAL']
            )

            if changed:
                # cam_brush_mode is True for Camera (index 2) and Tiled (index 3)
                if self.state.sim.current_view_option >= len(self.view_option_labels):
                    self.state.camera.cam_brush_mode = True
                else:
                    self.state.camera.cam_brush_mode = False

            # Physics tooltips checkbox
            _, self.state.preferences.physics_tooltips_enabled = imgui.checkbox(
                "Physics Tooltips",
                self.state.preferences.physics_tooltips_enabled
            )
            self._delayed_tooltip("Enable verbose tooltip and vector diagram for physics sliders.")

            # View Trail Arrows checkbox (renamed from Debug Arrows)
            _, self.state.preferences.debug_arrows = imgui.checkbox(
                "View Trail Arrows",
                self.state.preferences.debug_arrows
            )
            self._delayed_tooltip("Render a grid of arrows to help visualize canvas' vector field.")

            # Arrow sensitivity slider (only show when debug arrows enabled)
            if self.state.preferences.debug_arrows:
                imgui.indent(20)
                _, self.state.preferences.arrow_sensitivity = imgui.slider_float(
                    "Arrow Sensitivity",
                    self.state.preferences.arrow_sensitivity,
                    1.0, 20.0,
                    format="%.1f"
                )
                imgui.unindent(20)

            imgui.separator()

            # === Appearance section ===
            imgui.text("Appearance")

            # Brightness slider
            _, self.state.preferences.brightness = imgui.slider_float(
                "Brightness",
                self.state.preferences.brightness,
                0.0, 4.0,
                format="%.2f"
            )
            self._delayed_tooltip("Global brightness multiplier for the output.")

            # Exposure / Cheap Blur slider
            _, self.state.preferences.exposure = imgui.slider_float(
                "Exposure / Cheap Blur",
                self.state.preferences.exposure,
                0.0, 1.0,
                format="%.2f"
            )
            self._delayed_tooltip("Blend frames together for a cheap motion blur or set near 1 for a long exposure effect.")

        imgui.end()

        # Restore normal window background color if it was changed
        if recording_active:
            imgui.pop_style_color()

    def _get_key_combo(self, action: str, modifier: str = "") -> str:
        """
        Get a formatted key combination string for display.

        Args:
            action: The action name from keyboard_controls.json
            modifier: Optional modifier like "Ctrl+" or "Shift+"

        Returns:
            Formatted string like "Ctrl+C" or "WASD"
        """
        key = self.keybindings.get_key_display_name(action)
        if modifier:
            return f"{modifier}{key}"
        return key

    def render_controls_window(self):
        """Render the Controls help window (closeable)."""
        expanded, self.state.preferences.show_controls_window = imgui.begin("Controls", True)

        if expanded:
            imgui.text("Keyboard Controls")
            imgui.separator()

            # Camera movement keys
            w = self.keybindings.get_key_display_name("camera_forward")
            a = self.keybindings.get_key_display_name("camera_left")
            s = self.keybindings.get_key_display_name("camera_backward")
            d = self.keybindings.get_key_display_name("camera_right")
            imgui.bullet_text(f"{w}{a}{s}{d} - Move camera")

            q = self.keybindings.get_key_display_name("camera_out")
            e = self.keybindings.get_key_display_name("camera_in")
            imgui.bullet_text(f"{q}/{e} - Zoom out/in")

            imgui.bullet_text(f"Ctrl+{self.keybindings.get_key_display_name('copy_config_with_ctrl')} - Copy config to clipboard")
            imgui.bullet_text(f"Ctrl+{self.keybindings.get_key_display_name('paste_config_with_ctrl')} - Paste config from clipboard")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('randomize_mutations')} - Randomize rule seed")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('exit_keybinding')} - Exit application")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('record_screen')} - Toggle video recording")
            imgui.bullet_text("Shift+P - Take screenshot")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_pause')} - Pause/resume simulation")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('reset_keybinding')} - Reset particles to Initial Conditions")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_parameter_sweep')} - Toggle parameter sweeps")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_watercolor')} - Toggle watercolor mode")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('reload_shaders')} - Reload shaders")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_help')} - Show tutorial")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_mouse_mode')} - Toggle mouse mode")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('randomize_rules')} - Full reset (push zero rule + reset sim)")

            imgui.spacing()
            imgui.text("Mouse Controls")
            imgui.separator()

            imgui.text("Select Particle mode:")
            imgui.indent(20)
            imgui.bullet_text("Left click - Push active rule (select particle)")
            imgui.bullet_text("Right click - Pop active rule (revert)")
            imgui.unindent(20)

            imgui.text("Draw Trail mode:")
            imgui.indent(20)
            imgui.bullet_text("Click and drag - Draw trails on the canvas")
            imgui.unindent(20)

            imgui.spacing()
            imgui.text("Physics Slider Tips")
            imgui.separator()

            imgui.bullet_text("Right-click slider - Context menu to adjust range")
            imgui.bullet_text("Ctrl+click slider - Enter custom value directly")

        imgui.end()

    def render_parameter_sweeps_window(self):
        """Render the Parameter Sweeps help window (closeable)."""
        expanded, self.state.preferences.show_parameter_sweeps_window = imgui.begin("Parameter Sweeps", True)

        if expanded:
            imgui.text_wrapped(
                "Parameter sweeps let you vary physics settings across the screen, "
                "creating a gradient where each position uses different parameter values."
            )

            imgui.spacing()
            imgui.text("How to Use")
            imgui.separator()

            sweep_key = self.keybindings.get_key_display_name('toggle_parameter_sweep')
            imgui.bullet_text(f"Enable sweeps: Additional Settings -> Parameter Sweeps (or press {sweep_key})")
            imgui.bullet_text("Each parameter can sweep on X-axis, Y-axis, or by Cohort")
            imgui.bullet_text("The swept parameter will vary from slider_min to slider_max")
            imgui.bullet_text("Each slider has up/down buttons to its left which\nwiden/narrow the slider range.")
            imgui.bullet_text("Right click on sliders to manually set ranges")
            imgui.bullet_text("When using X and or Y sweeps, click anywhere on the canvas to\nset slider values. Then when you turn sweeps off,\neverywhere will behave like the region you clicked. ")
            imgui.bullet_text("Right click will temporarily disable sweeps allowing you to see\nthe effects of your slider values. Click anywhere to end the 'preview'.")

            imgui.spacing()
            imgui.text("Sweep Directions")
            imgui.separator()
            
            imgui.bullet_text("Left click for Normal, Right click for Inverse")
            imgui.bullet_text("Normal (->): Left/bottom = min, Right/top = max")
            imgui.bullet_text("Inverse (<-): Left/bottom = max, Right/top = min")
            imgui.bullet_text("Off: Parameter uses its slider value everywhere")

            imgui.spacing()
            imgui.text("Tips")
            imgui.separator()
            imgui.bullet_text("Some sliders have hard capped ranges, others are unbounded")
            imgui.text_wrapped(
                "Cohort sweeps vary parameters across particle groups rather than "
                "screen position."
            )
            imgui.spacing()
            imgui.text_wrapped(
                "Combine X and Y sweeps on different parameters to explore "
                "2D parameter spaces. For example, sweep Drag on X and "
                "Sensor Angle on Y to see how they interact. My most common "
                "pairs are Sensor Gain + Global force and Sensor Angle + Drag."
                "\nSensor gain and Global force often need to be changed together "
                "to keep energy ~constant. Sensor Angle + Drag is convenient "
                "because both ranges are limited and the full 2d domain can be"
                "viewed at once"
            )

        imgui.end()

    def render_tutorial_window(self):
        """Render the Tutorial help window (closeable)."""
        expanded, self.state.preferences.show_tutorial_window = imgui.begin("Tutorial", True)

        if expanded:
            imgui.text_wrapped(
                "Hive Explorer is like an interactive lava lamp. "
                "Thousands of particles interact resulting in a great variety of forms and patterns. "
                "Thumb through the presets in File->Load to see what is possible."
            )

            imgui.spacing()
            if imgui.collapsing_header("Basics", imgui.TreeNodeFlags_.default_open):
                imgui.text_wrapped("(see help->Controls for more)")
                # Camera movement keys
                w = self.keybindings.get_key_display_name("camera_forward")
                a = self.keybindings.get_key_display_name("camera_left")
                s = self.keybindings.get_key_display_name("camera_backward")
                d = self.keybindings.get_key_display_name("camera_right")
                imgui.bullet_text(f"Move the camera around with {w}{a}{s}{d}.")
                q = self.keybindings.get_key_display_name("camera_out")
                e = self.keybindings.get_key_display_name("camera_in")
                imgui.bullet_text(f"Zoom in or out with {q}/{e} or scroll wheel.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('reset_keybinding')} to reset the simulation.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_pause')} to toggle pause.")
                imgui.bullet_text("Click to draw trails or select particles.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_mouse_mode')} to toggle between drawing and selecting.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_help')} to toggle this Help window.")
            
            imgui.spacing()
            if imgui.collapsing_header("Rules"):
                imgui.text_wrapped(
                    "There is no fixed particle behavior in Hive Explorer. "
                    "Each particle has a 'Rule' which determines how it moves in response to nearby trails. "
                    "These rules can be mutated and evolved. In particle selection mouse mode, click on a particle to set it's rule as the 'active Rule'. "
                    "Now every particle will adopt that rule (with mutations). right click to undo setting a new target rule."
                )
            imgui.spacing()
            if imgui.collapsing_header("Save/Load"):
                copy_key = self.keybindings.get_key_display_name('copy_config_with_ctrl')
                paste_key = self.keybindings.get_key_display_name('paste_config_with_ctrl')
                imgui.text_wrapped(
                    "Create something you like? Save it as a new preset with File->Save"
                    "The active rule, current mutations, and everything on the physics panel will be restored when you load the save (Physics Sliders, Additional Settings, and Appearance) "
                    f"You can also press Ctrl-{copy_key} to copy a 'save string' to your clipboard, and Ctrl-{paste_key} to load a save string from the clipboard. "
                )

            imgui.spacing()
            if imgui.collapsing_header("Trails"):
                imgui.text_wrapped(
                    "Particles in hive explorer can't directly 'see' each other. "
                    "Instead, they interact by leaving pheremone trails as they move, like ants. "
                    "These trails accumulate on the 'Canvas' where particles can see them. Trails spread out and fade over time. "
                    "You can try writing your own pheremone trails to the canvas with 'Draw Trails' mouse mode (Preferences -> mouse mode)"
                )



            imgui.spacing()
            if imgui.collapsing_header("Mutations"):
                randomize_key = self.keybindings.get_key_display_name('randomize_rules')
                imgui.text_wrapped(
                    "Particles are grouped into 'Cohorts'. Each cohort shares a single mutation, so all the particles in a given cohort behave the same. "
                    "When the mutation Rate is greater than 0, different cohorts can behave differently, sometimes radically so. "
                    "The new rules generated by these mutations can also be selected as the active rule, so you can evolve particle behavior over many iterations. "
                    f"If you want to reset all the cohorts to random Rules, press {randomize_key}. (this can be undone with right click)"
                )

            imgui.spacing()
            if imgui.collapsing_header("Sliders"):
                imgui.text_wrapped(
                    "All sliders support Ctrl click to enter custom values. You can exceed slider range this way, "
                    "but it isn't always advisable. The sliders in the Physics Settings panel are special. Right click on them to set "
                    "custom ranges with the context menu. You can also vary their value across the canvas with parameter sweeps. See Help -> Parameter Sweeps for more."
                )

        imgui.end()

    def render_performance_window(self):
        """Render the Performance help window (closeable)."""
        expanded, self.state.preferences.show_performance_window = imgui.begin("Performance", True)

        if expanded:
            imgui.text_wrapped(
                "The options for World size, Physics update Frequency, and motion blur "
                "can significantly affect performance. World size and update frequency "
                "trade against each other so if you double one, halve the other for similar performance."
                "Motion blur gets more expensive with large worldsizes."
            )

            imgui.spacing()
            imgui.text("Example Setups")
            imgui.separator()

            imgui.bullet_text("x20 physics frequency with worldsize 0.3, motion blur every 4 frames")
            imgui.bullet_text("x5 physics frequency with worldsize 1.0, motion blur every frame")

            imgui.spacing()
            imgui.text_wrapped(
                "These run well on my 5060."
            )

        imgui.end()

    def render_video_recording_window(self):
        """Render the Screen Recording controls window (closeable)."""
        recording_active = self._display_info.get('recording_active', False)

        # Apply red tint when recording
        if recording_active:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

        expanded, self.show_video_recording_window = imgui.begin("Screen Recording", True)

        if expanded:
            record_key = self.keybindings.get_key_display_name('record_screen')
            if recording_active:
                imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "RECORDING IN PROGRESS")
                imgui.text(f"Press {record_key} to stop recording")
                imgui.separator()

            imgui.text(f"Press {record_key} to start/stop video recording")
            imgui.text(f"Press Shift+{record_key} to take a screenshot")
            imgui.spacing()

            # Video Length (in seconds) - converts to/from max_frames internally
            video_length_seconds = self.state.preferences.max_frames / 60.0
            changed, new_length = imgui.drag_float(
                'Video Length',
                video_length_seconds,
                v_speed=0.5,
                v_min=1.0,
                v_max=300.0,
                format="%.0f seconds"
            )
            if changed:
                self.state.preferences.max_frames = int(new_length * 60)
            self._delayed_tooltip("After Video reaches this length, the recording will be stopped")

            # Lock motion_blur_samples during recording
            if recording_active:
                imgui.begin_disabled()

            # Capture Physics Frequency / Screenshot samples
            current_hz = self.state.preferences.motion_blur_samples * 60
            _, self.state.preferences.motion_blur_samples = imgui.slider_int(
                'Capture Physics Frequency',
                self.state.preferences.motion_blur_samples,
                v_min=1,
                v_max=12,
                format=f"x%d ({current_hz}hz)"
            )
            self._delayed_tooltip("Physics steps per frame for video/screenshots.\nHigher values = smoother motion blur.\nAlso determines screenshot quality (samples blended together).")

            if recording_active:
                imgui.end_disabled()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.8, 0.0, 1.0),
                    "(Locked during recording)"
                )

            # Motion Blur checkbox (overrides preferences during recording)
            _, self.state.preferences.recording_motion_blur = imgui.checkbox(
                "Motion Blur (Recording)",
                self.state.preferences.recording_motion_blur
            )
            self._delayed_tooltip("Enable motion blur during video recording.\nThis setting overrides the Motion Blur checkbox in Preferences while recording.")

            # Blur Quality slider (only shown when recording motion blur is enabled)
            if self.state.preferences.recording_motion_blur:
                imgui.indent(20)
                # Custom format for blur quality
                blur_val = self.state.preferences.recording_blur_quality
                if blur_val == 1:
                    blur_format = "1 : Every Frame"
                else:
                    blur_format = f"{blur_val} : Every {blur_val} Frames"

                _, self.state.preferences.recording_blur_quality = imgui.slider_int(
                    "Blur Quality (Recording)",
                    self.state.preferences.recording_blur_quality,
                    1, 20,
                    format=blur_format
                )
                self._delayed_tooltip("Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance.\nThis setting overrides the Blur Quality slider in Preferences while recording.")
                imgui.unindent(20)

            # Downsample Resolution Factor (was Supersample Kernel Width)
            _, self.state.preferences.supersample_k = imgui.input_int('Downsample Resolution Factor', self.state.preferences.supersample_k)
            self._delayed_tooltip("Set to '2' to render a video at half resolution.")

            # Filename input
            _, self.state.preferences.filename_prefix = imgui.input_text(
                'Filename',
                self.state.preferences.filename_prefix,
                256
            )
            self._delayed_tooltip("Defaults to 'animation' if left empty. All filenames get timestamps appended")

        imgui.end()

        if recording_active:
            imgui.pop_style_color()

    def _delayed_tooltip(self, text: str):
        """Show tooltip with delay, requiring mouse to be stationary."""
        # HoveredFlags_.delay_normal provides medium delay, stationary provides "mouse must be still"
        if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.stationary):
            imgui.set_tooltip(text)

    def render_popup_modals(self):
        """Render popup modals (Save, Overwrite, Delete) - called regardless of sidebar visibility."""
        # Save popup modal
        if self.save_popup_open:
            imgui.open_popup("Save Config")

        if imgui.begin_popup_modal("Save Config", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text("Enter filename (without extension):")
            _, self.save_filename_buffer = imgui.input_text(
                "##filename",
                self.save_filename_buffer,
            )

            imgui.separator()
            if imgui.button("Save", imgui.ImVec2(120, 0)):
                if self.save_filename_buffer.strip():
                    filename = self.save_filename_buffer.strip()
                    filepath = self.configs_dir / f"{filename}.json"
                    if filepath.exists():
                        # File exists, need overwrite confirmation
                        # Close save popup first, then open overwrite popup
                        self.overwrite_confirm_filename = filename
                        self.save_popup_open = False
                        imgui.close_current_popup()
                    else:
                        # File doesn't exist, save directly
                        self._save_filename = filename
                        self._request_save_file = True
                        self.save_popup_open = False
                        imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.save_popup_open = False
                imgui.close_current_popup()
            imgui.end_popup()

        # Overwrite confirmation popup
        if self.overwrite_confirm_filename:
            imgui.open_popup("Overwrite?")

        if imgui.begin_popup_modal("Overwrite?", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text(f"File '{self.overwrite_confirm_filename}.json' already exists.")
            imgui.text("Do you want to overwrite it?")
            imgui.separator()
            if imgui.button("Overwrite", imgui.ImVec2(120, 0)):
                self._save_filename = self.overwrite_confirm_filename
                self._request_save_file = True
                self.overwrite_confirm_filename = None
                self.save_popup_open = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.overwrite_confirm_filename = None
                imgui.close_current_popup()
            imgui.end_popup()

        # Delete confirmation popup
        if self.delete_confirm_filename:
            imgui.open_popup("Delete Config?")

        if imgui.begin_popup_modal("Delete Config?", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text(f"Are you sure you want to delete '{self.delete_confirm_filename}.json'?")
            imgui.separator()
            if imgui.button("Delete", imgui.ImVec2(120, 0)):
                self._delete_filename = self.delete_confirm_filename
                self._request_delete_file = True
                self.delete_confirm_filename = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                self.delete_confirm_filename = None
                imgui.close_current_popup()
            imgui.end_popup()

    def render_physics_settings_window(self):
        """Render the Physics Settings window with sliders."""
        # Apply bluish background when in sweep preview mode (waiting for click to restore sweeps)
        if self.state.sim.sweep_preview_pending_restore:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.15, 0.20, 0.35, 0.94))
            imgui.push_style_color(imgui.Col_.title_bg_active, imgui.ImVec4(0.20, 0.30, 0.50, 1.0))

        # No p_open parameter - window is uncloseable
        imgui.begin('Physics Settings', flags=imgui.WindowFlags_.menu_bar)

        # Track if any physics menu is open
        physics_any_menu_open_this_frame = False

        # Multi-load mode: different rendering
        if self.state.multi_load.multi_load_enabled:
            # Render multi-load specific UI
            if imgui.begin_menu_bar():
                # Track all open menu rectangles separately
                physics_menu_rectangles = []

                # Start with the menu bar itself
                physics_menu_bar_min = imgui.get_window_pos()
                physics_menu_bar_size = imgui.get_window_size()
                physics_menu_rectangles.append((physics_menu_bar_min.x, physics_menu_bar_min.y,
                                               physics_menu_bar_min.x + physics_menu_bar_size.x,
                                               physics_menu_bar_min.y + physics_menu_bar_size.y))

                if imgui.begin_menu("Multi-Load Settings", not self.force_close_physics_menus):
                    physics_any_menu_open_this_frame = True
                    # Add this menu's bounding box to the list
                    multi_load_menu_min = imgui.get_window_pos()
                    multi_load_menu_size = imgui.get_window_size()
                    physics_menu_rectangles.append((multi_load_menu_min.x, multi_load_menu_min.y,
                                                   multi_load_menu_min.x + multi_load_menu_size.x,
                                                   multi_load_menu_min.y + multi_load_menu_size.y))

                    # Particle Assignment combo
                    assignment_options = ["Random", "Cohorts"]
                    current_idx = 0 if self.state.multi_load.assignment_mode == "Random" else 1
                    imgui.set_next_item_width(150)
                    changed, new_idx = imgui.combo("Particle Assignment", current_idx, assignment_options)
                    if changed:
                        self.state.multi_load.assignment_mode = assignment_options[new_idx]
                    self._delayed_tooltip("Random: each particle randomly assigned\nCohorts: particles grouped by cohort")

                    _, self.state.multi_load.per_config_initial_conditions = imgui.checkbox(
                        "Per-config Initial Conditions", self.state.multi_load.per_config_initial_conditions)
                    self._delayed_tooltip("Each config uses its own initial conditions")

                    _, self.state.multi_load.per_config_cohorts = imgui.checkbox(
                        "Per-config Cohorts", self.state.multi_load.per_config_cohorts)
                    self._delayed_tooltip("Each config uses its own cohort count")

                    _, self.state.multi_load.per_config_hazard_rate = imgui.checkbox(
                        "Per-config Hazard Rate", self.state.multi_load.per_config_hazard_rate)
                    self._delayed_tooltip("Each config uses its own hazard rate setting")

                    imgui.end_menu()

                # Simplified Additional Settings (some items greyed out)
                if imgui.begin_menu("Additional Settings", not self.force_close_physics_menus):
                    physics_any_menu_open_this_frame = True
                    # Add this menu's bounding box to the list
                    additional_menu_min = imgui.get_window_pos()
                    additional_menu_size = imgui.get_window_size()
                    physics_menu_rectangles.append((additional_menu_min.x, additional_menu_min.y,
                                                   additional_menu_min.x + additional_menu_size.x,
                                                   additional_menu_min.y + additional_menu_size.y))

                    # Boundary Conditions
                    boundary_options = ["Bounce", "Reset", "Wrap"]
                    imgui.set_next_item_width(100)
                    if imgui.begin_combo("Boundary Conditions", boundary_options[self.state.sim.boundary_conditions]):
                        for i, option in enumerate(boundary_options):
                            if imgui.selectable(option, self.state.sim.boundary_conditions == i)[0]:
                                self.state.sim.boundary_conditions = i
                        imgui.end_combo()

                    # Initial Conditions (greyed if per-config)
                    if self.state.multi_load.per_config_initial_conditions:
                        imgui.begin_disabled()
                    initial_options = ["Grid", "Random", "Ring"]
                    imgui.set_next_item_width(100)
                    if imgui.begin_combo("Initial Conditions", initial_options[self.state.sim.initial_conditions]):
                        for i, option in enumerate(initial_options):
                            if imgui.selectable(option, self.state.sim.initial_conditions == i)[0]:
                                self.state.sim.initial_conditions = i
                        imgui.end_combo()
                    if self.state.multi_load.per_config_initial_conditions:
                        imgui.end_disabled()

                    # Cohorts (greyed if per-config)
                    if self.state.multi_load.per_config_cohorts:
                        imgui.begin_disabled()
                    imgui.set_next_item_width(100)
                    _, self.state.sim.num_cohorts = imgui.slider_int("Number of Cohorts", self.state.sim.num_cohorts, 1, 144)
                    if self.state.multi_load.per_config_cohorts:
                        imgui.end_disabled()

                    # Hazard Rate (conditional on per-config setting)
                    if self.state.multi_load.per_config_hazard_rate:
                        imgui.begin_disabled()
                    imgui.set_next_item_width(100)
                    _, self.state.sim.HAZARD_RATE = imgui.slider_float("Hazard Rate", self.state.sim.HAZARD_RATE, 0.0, 0.05, "%.4f")
                    if self.state.multi_load.per_config_hazard_rate:
                        imgui.end_disabled()
                    self._delayed_tooltip("Probability per frame that particles reset to initial conditions")

                    # Disable these options in multi-load (per-config settings)
                    imgui.begin_disabled()
                    imgui.checkbox("Disable Symmetry", False)
                    imgui.checkbox("Absolute Orientation", False)
                    imgui.end_disabled()
                    self._delayed_tooltip("Per-config settings in Multi-Load mode")

                    # Parameter Sweeps (disabled)
                    imgui.begin_disabled()
                    imgui.checkbox("Parameter Sweeps", False)
                    imgui.end_disabled()
                    self._delayed_tooltip("Disabled in Multi-Load mode")

                    imgui.end_menu()

                # Appearance (unchanged, copy from normal mode)
                if imgui.begin_menu("Appearance", not self.force_close_physics_menus):
                    physics_any_menu_open_this_frame = True
                    # Add this menu's bounding box to the list
                    appearance_menu_min = imgui.get_window_pos()
                    appearance_menu_size = imgui.get_window_size()
                    physics_menu_rectangles.append((appearance_menu_min.x, appearance_menu_min.y,
                                                   appearance_menu_min.x + appearance_menu_size.x,
                                                   appearance_menu_min.y + appearance_menu_size.y))

                    _, self.state.sim.color_by_cohort = imgui.checkbox("Color by Cohort", self.state.sim.color_by_cohort)
                    if not self.state.sim.color_by_cohort:
                        imgui.set_next_item_width(100)
                        _, self.state.sim.hue_sensitivity = imgui.slider_float("Hue Sensitivity", self.state.sim.hue_sensitivity, -1.0, 1.0)
                    _, self.state.sim.watercolor_mode = imgui.checkbox("Watercolor Mode", self.state.sim.watercolor_mode)
                    if self.state.sim.watercolor_mode:
                        imgui.set_next_item_width(100)
                        _, self.state.sim.ink_weight = imgui.slider_float("Ink Weight", self.state.sim.ink_weight, 0.0, 4.0)
                    emboss_options = ["Off", "Canvas (Trails)", "Brush (Particles)"]
                    imgui.set_next_item_width(150)
                    if imgui.begin_combo("Emboss Mode", emboss_options[self.state.sim.emboss_mode]):
                        for i, option in enumerate(emboss_options):
                            if imgui.selectable(option, self.state.sim.emboss_mode == i)[0]:
                                self.state.sim.emboss_mode = i
                        imgui.end_combo()
                    if self.state.sim.emboss_mode != 0:
                        imgui.set_next_item_width(100)
                        _, self.state.sim.emboss_intensity = imgui.slider_float("Emboss Intensity", self.state.sim.emboss_intensity, -1.0, 1.0)
                        imgui.set_next_item_width(100)
                        _, self.state.sim.emboss_smoothness = imgui.slider_float("Emboss Smoothness", self.state.sim.emboss_smoothness, 0.001, 1.0)
                    imgui.end_menu()

                # After all menus: check mouse distance from all menu rectangles
                # Find the minimum distance to any rectangle
                if self.physics_menu_bar_has_open_menu and not self.save_popup_open:
                    mouse_pos = imgui.get_mouse_pos()

                    # Calculate minimum distance to any menu rectangle
                    min_distance = float('inf')
                    for min_x, min_y, max_x, max_y in physics_menu_rectangles:
                        dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                        dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                        distance = (dx * dx + dy * dy) ** 0.5
                        min_distance = min(min_distance, distance)

                    # If mouse is too far away from all rectangles, signal to close menus
                    if min_distance > self.state.preferences.menu_close_threshold:
                        self.force_close_physics_menus = True

                imgui.end_menu_bar()

            # Update physics menu tracking state
            self.physics_menu_bar_has_open_menu = physics_any_menu_open_this_frame
            # Reset force close flag after processing
            if self.force_close_physics_menus and not physics_any_menu_open_this_frame:
                self.force_close_physics_menus = False

            # Multi-load controls
            imgui.text("Multi-Load Controls")
            imgui.separator()

            # Get config count from service
            config_count = self.multi_load_service.get_config_count() if self.multi_load_service else 0

            _, self.state.multi_load.simultaneous_configs = imgui.slider_float(
                "Simultaneous Configs", self.state.multi_load.simultaneous_configs, 0.0, float(max(1, config_count-.001)))
            _, self.state.multi_load.progression_pace = imgui.slider_float(
                "Progression Pace", self.state.multi_load.progression_pace, 0.0, 1.0)

            # Sync current progress from service (for auto-advancement display)
            if self.multi_load_service:
                current_progress_value = self.multi_load_service.current_progress
            else:
                current_progress_value = self.state.multi_load.current_progress

            changed, new_progress = imgui.slider_float(
                "Current Progress", current_progress_value, 0.0, 1.0)
            if changed and self.multi_load_service:
                # User manually changed the slider - update service directly
                self.multi_load_service.set_progress(new_progress)
            # Always sync state from service for next frame
            if self.multi_load_service:
                self.state.multi_load.current_progress = self.multi_load_service.current_progress

            imgui.separator()
            imgui.text(f"Loaded Configurations ({config_count}/64)")
            imgui.separator()

            if config_count == 0:
                imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "No configs loaded")
                imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Use File -> Load to add")
            else:
                # Render config list with remove buttons
                for i in range(config_count):
                    filename = self.multi_load_service.get_filename(i)
                    if filename:
                        # Config name
                        imgui.text(f"{i+1}. {filename}")
                        imgui.same_line()
                        # Remove button (aligned to right)
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
                        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                        if imgui.small_button(f"Remove##{i}"):
                            self.multi_load_service.remove_config(i)
                        imgui.pop_style_color(2)

            imgui.end()
            if self.state.sim.sweep_preview_pending_restore:
                imgui.pop_style_color(2)
            return

        # Normal mode: render sliders and settings
        # Additional Settings menu bar
        if imgui.begin_menu_bar():
            # Track all open menu rectangles separately
            physics_menu_rectangles = []

            # Start with the menu bar itself
            physics_menu_bar_min = imgui.get_window_pos()
            physics_menu_bar_size = imgui.get_window_size()
            physics_menu_rectangles.append((physics_menu_bar_min.x, physics_menu_bar_min.y,
                                           physics_menu_bar_min.x + physics_menu_bar_size.x,
                                           physics_menu_bar_min.y + physics_menu_bar_size.y))

            if imgui.begin_menu("Additional Settings", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                additional_settings_menu_min = imgui.get_window_pos()
                additional_settings_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((additional_settings_menu_min.x, additional_settings_menu_min.y,
                                               additional_settings_menu_min.x + additional_settings_menu_size.x,
                                               additional_settings_menu_min.y + additional_settings_menu_size.y))

                # Boundary Conditions (with per-option tooltips)
                boundary_options = ["Bounce", "Reset", "Wrap"]
                boundary_tooltips = [
                    "Particles bounce off the edges of the canvas",
                    "Particles are reset to their initial conditions when leaving the canvas",
                    "Particles wrap seamlessly to the other side of the canvas"
                ]
                imgui.set_next_item_width(100)
                if imgui.begin_combo("Boundary Conditions", boundary_options[self.state.sim.boundary_conditions]):
                    for i, option in enumerate(boundary_options):
                        is_selected = (self.state.sim.boundary_conditions == i)
                        if imgui.selectable(option, is_selected)[0]:
                            self.state.sim.boundary_conditions = i
                        self._delayed_tooltip(boundary_tooltips[i])
                        if is_selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

                # Initial Conditions (with per-option tooltips)
                initial_options = ["Grid", "Random", "Ring"]
                initial_tooltips = [
                    "Particles start in a grid, organized by cohort",
                    "Particles are spread uniformly across the canvas",
                    "Particles start distributed around a circle, organized by cohort"
                ]
                imgui.set_next_item_width(100)
                if imgui.begin_combo("Initial Conditions", initial_options[self.state.sim.initial_conditions]):
                    for i, option in enumerate(initial_options):
                        is_selected = (self.state.sim.initial_conditions == i)
                        if imgui.selectable(option, is_selected)[0]:
                            self.state.sim.initial_conditions = i
                        self._delayed_tooltip(initial_tooltips[i])
                        if is_selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

                # Number of Cohorts
                imgui.set_next_item_width(100)
                _, self.state.sim.num_cohorts = imgui.slider_int(
                    "Number of Cohorts",
                    self.state.sim.num_cohorts,
                    1, 144
                )
                self._delayed_tooltip("Each particle is assigned to a cohort. Each cohort shares behavior\nand there can be mutations between different cohorts.")

                imgui.separator()

                # Disable Symmetry
                _, self.state.sim.DISABLE_SYMMETRY = imgui.checkbox(
                    "Disable Symmetry",
                    self.state.sim.DISABLE_SYMMETRY
                )
                self._delayed_tooltip("Allow particles to display \"right / left handed\" behavior,\nleading to clockwise/counterclockwise bias.\nTurn it on to see why we go through trouble\nof calculating \"mirror world\" behavior in entity_update.glsl")

                # Absolute Orientation (combo box with 3 modes)
                combo_items = ["Off", "Y axis", "Radial"]
                clicked, current = imgui.combo(
                    "Absolute Orientation",
                    self.state.sim.ABSOLUTE_ORIENTATION,
                    combo_items
                )
                if clicked:
                    self.state.sim.ABSOLUTE_ORIENTATION = current
                self._delayed_tooltip("What direction are particles 'facing'? Which way is 'up'?\nOff: use particle velocity\nY axis: align to y axis\nRadial: align to center of canvas")
                # Orientation Mix (only visible if Absolute Orientation != Off)
                if self.state.sim.ABSOLUTE_ORIENTATION != 0:
                    imgui.set_next_item_width(100)
                    _, self.state.sim.ORIENTATION_MIX = imgui.slider_float(
                        "Orientation Mix",
                        self.state.sim.ORIENTATION_MIX,
                        0.0, 1.0,
                        "%.2f"
                    )
                    self._delayed_tooltip("Blend factor for orientation calculations (0.0 = velocity only, 1.0 = full absolute orientation)")

                imgui.separator()

                # Parameter Sweeps toggle (moved from Extras menu)
                _, self.state.sim.parameter_sweeps_enabled = imgui.checkbox(
                    "Parameter Sweeps",
                    self.state.sim.parameter_sweeps_enabled
                )
                sweep_key = self.keybindings.get_key_display_name('toggle_parameter_sweep')
                self._delayed_tooltip(f"Enable parameter sweeps to vary physics across the canvas.\nPress {sweep_key} to toggle. See Help -> Parameter Sweeps for details.")

                imgui.end_menu()

            # Appearance menu
            if imgui.begin_menu("Appearance", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                appearance_settings_menu_min = imgui.get_window_pos()
                appearance_settings_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((appearance_settings_menu_min.x, appearance_settings_menu_min.y,
                                               appearance_settings_menu_min.x + appearance_settings_menu_size.x,
                                               appearance_settings_menu_min.y + appearance_settings_menu_size.y))

                # Color by cohort checkbox
                _, self.state.sim.color_by_cohort = imgui.checkbox(
                    "Color by Cohort",
                    self.state.sim.color_by_cohort
                )
                self._delayed_tooltip("Colors particles based on their cohort assignment\nrather than their behavior.")

                # Hue Sensitivity (only if not color by cohort)
                if not self.state.sim.color_by_cohort:
                    _, self.state.sim.hue_sensitivity = imgui.slider_float(
                        "Hue Sensitivity", self.state.sim.hue_sensitivity, -1.0, 1.0
                    )
                    self._delayed_tooltip("Controls color variation based on particle velocity.")

                imgui.separator()

                # Watercolor Mode checkbox
                _, self.state.sim.watercolor_mode = imgui.checkbox(
                    "Watercolor Mode (V)",
                    self.state.sim.watercolor_mode
                )

                # Ink Weight slider (only in watercolor mode, placed right after checkbox)
                if self.state.sim.watercolor_mode:
                    _, self.state.sim.ink_weight = imgui.slider_float(
                        "Ink Weight", self.state.sim.ink_weight, 0.0, 4.0
                    )
                    self._delayed_tooltip("Controls optical density in watercolor mode.\nHigher values = darker/more opaque.")
                self._delayed_tooltip("Enable watercolor rendering effect.")

                imgui.separator()

                # Emboss mode combo box
                emboss_options = ["Off", "Canvas (Trails)", "Brush (Particles)"]
                _, self.state.sim.emboss_mode = imgui.combo(
                    "Emboss", self.state.sim.emboss_mode, emboss_options
                )
                self._delayed_tooltip("Calculate some fake 3D lighting\nby treating (otherwise unused) particle\ndensity as a heightmap.")

                # Emboss sliders only visible when mode is not Off
                if self.state.sim.emboss_mode != 0:
                    # Emboss Intensity slider
                    _, self.state.sim.emboss_intensity = imgui.slider_float(
                        "Emboss Intensity", self.state.sim.emboss_intensity, -1.0, 1.0
                    )
                    self._delayed_tooltip("Intensity of emboss lighting effect. Negative values invert.")

                    # Emboss Smoothness slider
                    _, self.state.sim.emboss_smoothness = imgui.slider_float(
                        "Emboss Smoothness", self.state.sim.emboss_smoothness, 0.001, 1.0
                    )
                    self._delayed_tooltip("Controls the smoothness of emboss sampling.")

                imgui.end_menu()

            # After all menus: check mouse distance from all menu rectangles
            # Find the minimum distance to any rectangle
            if self.physics_menu_bar_has_open_menu and not self.save_popup_open:
                mouse_pos = imgui.get_mouse_pos()

                # Calculate minimum distance to any menu rectangle
                min_distance = float('inf')
                for min_x, min_y, max_x, max_y in physics_menu_rectangles:
                    dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                    dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # If mouse is too far away from all rectangles, signal to close menus
                if min_distance > self.state.preferences.menu_close_threshold:
                    self.force_close_physics_menus = True

            imgui.end_menu_bar()

        # Update physics menu tracking state
        self.physics_menu_bar_has_open_menu = physics_any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_physics_menus and not physics_any_menu_open_this_frame:
            self.force_close_physics_menus = False

        # Display currently open project
        imgui.text(f"Project: {self.currently_open_project}")
        imgui.separator()

        # === Basics Group (Trail sensors and rule mutation) ===
        imgui.set_next_item_open(self.state.preferences.physics_group_basics)
        basics_open = imgui.collapsing_header("Basics - Trail sensors and rule mutation")
        # Update state to match actual header state (handles user clicks)
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_basics = basics_open
        if basics_open:
            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("SENSOR_GAIN")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("SENSOR_GAIN", "Sensor Gain", self.state.sim.SENSOR_GAIN, 0.0, 5.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.SENSOR_GAIN = self.slider_float_with_range_menu(
                label="Sensor Gain",
                param_name="SENSOR_GAIN",
                value=self.state.sim.SENSOR_GAIN,
                default_min=0,
                default_max=5.0,
            )
            self.render_custom_tooltip("Sensor Gain",
                "Determines how strongly particles respond to sensor input. Higher values make particles more reactive to the trails they sense on the Canvas.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("SENSOR_ANGLE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("SENSOR_ANGLE", "Sensor Angle", self.state.sim.SENSOR_ANGLE, -1.0, 1.0, hard_min=-1.0, hard_max=1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.SENSOR_ANGLE = self.slider_float_with_range_menu(
                label="Sensor Angle",
                param_name="SENSOR_ANGLE",
                value=self.state.sim.SENSOR_ANGLE,
                default_min=-1.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Sensor Angle",
                "Sets the angular offset of particle sensors from their forward direction. Determines whether particles are 'looking ahead' or 'looking behind'.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("SENSOR_DISTANCE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("SENSOR_DISTANCE", "Sensor Distance", self.state.sim.SENSOR_DISTANCE, 0.0, 4.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.SENSOR_DISTANCE = self.slider_float_with_range_menu(
                label="Sensor Distance",
                param_name="SENSOR_DISTANCE",
                value=self.state.sim.SENSOR_DISTANCE,
                default_min=0.0,
                default_max=4.0,
            )
            self.render_custom_tooltip("Sensor Distance",
                "Determines distance between a particle's center and where it reads the trail information from Canvas. Longer distances tend to create larger scale patterns.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("MUTATION_SCALE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("MUTATION_SCALE", "Mutation Scale", self.state.sim.MUTATION_SCALE, -0.5, 0.5)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.MUTATION_SCALE = self.slider_float_with_range_menu(
                label="Mutation Scale",
                param_name="MUTATION_SCALE",
                value=self.state.sim.MUTATION_SCALE,
                default_min=-.5,
                default_max=.5,
            )
            self.render_custom_tooltip("Mutation Scale",
                "Controls the size of the random mutations applied to a rule when a new particle is clicked. At 0, every particle will behave exactly like the selected particle.")

        # === Forces Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_forces)
        forces_open = imgui.collapsing_header("Forces")
        # Update state to match actual header state (handles user clicks)
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_forces = forces_open
        if forces_open:
            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("GLOBAL_FORCE_MULT")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("GLOBAL_FORCE_MULT", "Global Force Mult", self.state.sim.GLOBAL_FORCE_MULT, 0.0, 2.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.GLOBAL_FORCE_MULT = self.slider_float_with_range_menu(
                label="Global Force Mult",
                param_name="GLOBAL_FORCE_MULT",
                value=self.state.sim.GLOBAL_FORCE_MULT,
                default_min=0.0,
                default_max=2.0,
            )
            self.render_custom_tooltip("Global Force Mult",
                "Scales axial and lateral forces applied to particles, and scales strafe power. Often tuned in the opposite direction to Sensor Gain and Drag to offset exploding/vanishing particle speed.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("DRAG")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("DRAG", "Drag", self.state.sim.DRAG, -1.0, 1.0, hard_min=-1.0, hard_max=1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.DRAG = self.slider_float_with_range_menu(
                label="Drag",
                param_name="DRAG",
                value=self.state.sim.DRAG,
                default_min=-1.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Drag",
                "Each physics update, particle velocity is multiplied by drag like so:   vel = vel*drag + forces; So drag less than 1 means particles are being slowed down. Powerful (<0.5) drag values can prevent energetic systems from 'blowing up'")

        # === Advanced Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_advanced)
        advanced_open = imgui.collapsing_header("Advanced")
        # Update state to match actual header state (handles user clicks)
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_advanced = advanced_open
        if advanced_open:
            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("AXIAL_FORCE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("AXIAL_FORCE", "Axial Force", self.state.sim.AXIAL_FORCE, -1.0, 1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.AXIAL_FORCE = self.slider_float_with_range_menu(
                label="Axial Force",
                param_name="AXIAL_FORCE",
                value=self.state.sim.AXIAL_FORCE,
                default_min=-1.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Axial Force",
                "Controls the strength of forces applied parallel to the direction of travel: acceleration and braking")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("LATERAL_FORCE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("LATERAL_FORCE", "Lateral Force", self.state.sim.LATERAL_FORCE, -1.0, 1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.LATERAL_FORCE = self.slider_float_with_range_menu(
                label="Lateral Force",
                param_name="LATERAL_FORCE",
                value=self.state.sim.LATERAL_FORCE,
                default_min=-1.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Lateral Force",
                "Controls the strength of forces applied perpendicular to the direction of travel: turning left and right.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("STRAFE_POWER")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("STRAFE_POWER", "Strafe Power", self.state.sim.STRAFE_POWER, 0.0, 0.5)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.STRAFE_POWER = self.slider_float_with_range_menu(
                label="Strafe Power",
                param_name="STRAFE_POWER",
                value=self.state.sim.STRAFE_POWER,
                default_min=0.0,
                default_max=0.5,
            )
            self.render_custom_tooltip("Strafe Power",
                "Controls particle movement without applying forces to velocity. 'Strafe' is a vector added directly to position each frame, like a little hop. Strafe power scales with Axial, Lateral, and Global force multipliers.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("TRAIL_PERSISTENCE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("TRAIL_PERSISTENCE", "Trail Persistence", self.state.sim.TRAIL_PERSISTENCE, 0.0, 1.0, hard_min=0.0, hard_max=1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.TRAIL_PERSISTENCE = self.slider_float_with_range_menu(
                label="Trail Persistence",
                param_name="TRAIL_PERSISTENCE",
                value=self.state.sim.TRAIL_PERSISTENCE,
                default_min=0.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Trail Persistence",
                "Controls how long particle trails remain visible. Higher values create longer-lasting trails, lower values make trails fade quickly. Values close to 1.0 tend to create 'sharper' more stable patterns. ")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("TRAIL_DIFFUSION")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("TRAIL_DIFFUSION", "Trail Diffusion", self.state.sim.TRAIL_DIFFUSION, 0.0, 1.0, hard_min=0.0, hard_max=1.0)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.TRAIL_DIFFUSION = self.slider_float_with_range_menu(
                label="Trail Diffusion",
                param_name="TRAIL_DIFFUSION",
                value=self.state.sim.TRAIL_DIFFUSION,
                default_min=0.0,
                default_max=1.0,
            )
            self.render_custom_tooltip("Trail Diffusion",
                "Controls how quickly particle trails spread out and blend together.")

            if self.state.sim.parameter_sweeps_enabled:
                self.render_sweep_buttons("HAZARD_RATE")
                imgui.same_line(spacing=2)
                self.render_range_adjust_buttons("HAZARD_RATE", "Hazard Rate", self.state.sim.HAZARD_RATE, 0.0, 0.05, hard_min=0.0, hard_max=0.05)
                imgui.same_line(spacing=8)
                imgui.set_next_item_width(80)

            _, self.state.sim.HAZARD_RATE = self.slider_float_with_range_menu(
                label="Hazard Rate",
                param_name="HAZARD_RATE",
                value=self.state.sim.HAZARD_RATE,
                default_min=0.0,
                default_max=0.05,
            )
            self.render_custom_tooltip("Hazard Rate",
                "Probability per frame that particles reset to initial conditions. Gives particles a probabalistic 'lifetime' after which they reset.")

        imgui.separator()

        # Render the tooltip if window is hovered
        self.render_physics_tooltip()

        imgui.end()

        # Pop sweep preview style colors (pushed before imgui.begin)
        if self.state.sim.sweep_preview_pending_restore:
            imgui.pop_style_color(2)

    def render_history_window(self):
        """Render rule history window with preview."""
        imgui.begin("Rule History")

        rule_history = self._display_info.get('rule_history', [])

        if not rule_history:
            imgui.text_colored(imgui.ImVec4(1.0, 0.5, 0.5, 1.0), "No rules in history")
            imgui.end()
            return

        # Sync metadata with rule history
        # When adding new rules, append new labels
        while len(self.history_window_labels) < len(rule_history):
            self.history_window_labels.append(self._generate_rule_label())
        # When removing old rules (from beginning), remove old labels (from beginning)
        while len(self.history_window_labels) > len(rule_history):
            self.history_window_labels.pop(0)

        # Determine how many rules to show (hide topmost if previewing)
        num_rules_to_show = len(rule_history)
        if self.currently_previewing_index is not None:
            # Previewing - hide the topmost element (it's the preview copy)
            num_rules_to_show -= 1

        # Render rules (newest first, but skip the preview if active)
        hovered_this_frame = None

        for i in range(num_rules_to_show - 1, -1, -1):
            # Get jersey number and colors
            jersey_number, color1_rgb, color2_rgb = self.history_window_labels[i]
            color1 = self._parse_rgb_color(color1_rgb)
            color2 = self._parse_rgb_color(color2_rgb)

            # Extract digits from jersey number
            digit1 = jersey_number // 10
            digit2 = jersey_number % 10

            # Render colored digits
            imgui.text_colored(imgui.ImVec4(*color1), str(digit1))
            imgui.same_line(spacing=0)
            imgui.text_colored(imgui.ImVec4(*color2), str(digit2))
            imgui.same_line(spacing=2)

            # Invisible selectable for click/hover detection
            clicked, _ = imgui.selectable(
                f"##{i}",
                False,
                imgui.SelectableFlags_.none,
                imgui.ImVec2(10, 0)  # Small width just for the hitbox
            )

            if imgui.is_item_hovered():
                hovered_this_frame = i

            # X button
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            if imgui.small_button(f"X##history_{i}"):
                self._request_delete_history_rule = True
                self._history_preview_index = i
            imgui.pop_style_color(2)

            if imgui.is_item_hovered():
                hovered_this_frame = i

            if clicked:
                self._request_load_history_rule = True
                self._history_preview_index = i

        # Handle preview state changes
        if hovered_this_frame != self.currently_previewing_index:
            if self.currently_previewing_index is not None:
                self._request_clear_history_preview = True

            if hovered_this_frame is not None:
                self._request_preview_history_rule = True
                self._history_preview_index = hovered_this_frame
                self.currently_previewing_index = hovered_this_frame
            else:
                self.currently_previewing_index = None

        imgui.end()

    def update_tooltip_texture(self):
        """Render the tooltip graphic to texture using shader."""
        # Set time uniform for animations
        current_time = time.time() - self.tooltip_start_time
        self.tooltip_program['time'] = current_time * 3.0  # Speed up animation a bit

        # Set sensor angle (raw value, not normalized)
        self.tooltip_program['SENSOR_ANGLE'] = self.state.sim.SENSOR_ANGLE

        # Set MODE bools based on which slider is hovered
        self.tooltip_program['AXIAL_MODE'] = (self.last_hovered_slider == "Axial Force")
        self.tooltip_program['LATERAL_MODE'] = (self.last_hovered_slider == "Lateral Force")
        self.tooltip_program['SENSOR_MODE'] = (self.last_hovered_slider == "Sensor Gain")
        self.tooltip_program['DRAG_MODE'] = (self.last_hovered_slider == "Drag")
        self.tooltip_program['ANGLE_MODE'] = (self.last_hovered_slider == "Sensor Angle")
        self.tooltip_program['DISTANCE_MODE'] = (self.last_hovered_slider == "Sensor Distance")
        self.tooltip_program['TRAIL_MODE'] = (self.last_hovered_slider == "Trail Persistence")
        self.tooltip_program['DIFFUSION_MODE'] = (self.last_hovered_slider == "Trail Diffusion")
        self.tooltip_program['GLOBAL_MODE'] = (self.last_hovered_slider == "Global Force Mult")
        self.tooltip_program['STRAFE_MODE'] = (self.last_hovered_slider == "Strafe Power")
        self.tooltip_program['MUTATION_MODE'] = (self.last_hovered_slider == "Mutation Scale")

        # Render to framebuffer
        self.tooltip_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tooltip_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        self.ctx.screen.use()  # Return to default framebuffer

    def render_custom_tooltip(self, label: str, description: str):
        """Render a custom tooltip anchored to the right edge of a window.

        Args:
            label: The label of the slider
            description: Description text to display in the tooltip
        """
        # Track which slider is currently hovered
        if imgui.is_item_hovered():
            self.last_hovered_slider = label
            self.last_hovered_description = description

        # Track if any item is being actively manipulated (dragged)
        if imgui.is_item_active():
            self.physics_window_interaction = True

    def render_physics_tooltip(self):
        """Render the tooltip if mouse is over the Physics Settings window."""
        # Early exit if tooltips are disabled
        if not self.state.preferences.physics_tooltips_enabled:
            self.last_hovered_slider = None
            self.physics_window_interaction = False
            return

        # Check if physics settings window is hovered or if we're actively interacting with it
        physics_window_hovered = imgui.is_window_hovered()

        # First, check if we should show the tooltip at all
        # We need to render it at least once to check if IT is hovered
        should_show = (physics_window_hovered or
                      self.physics_window_interaction or
                      self.last_hovered_slider is not None)

        if not should_show:
            self.last_hovered_slider = None
            self.physics_window_interaction = False
            return

        if self.last_hovered_slider is None:
            self.physics_window_interaction = False
            return

        # Update the tooltip texture with current slider values
        self.update_tooltip_texture()

        # Get the position and size of the anchor window
        window_pos = imgui.get_window_pos()
        window_size = imgui.get_window_size()

        # Calculate tooltip position (right edge of the anchor window)
        tooltip_x = window_pos.x + window_size.x
        tooltip_y = window_pos.y

        # Set next window position
        imgui.set_next_window_pos(imgui.ImVec2(tooltip_x, tooltip_y))

        # Begin a borderless, no-move, no-focus tooltip window
        # Note: no_focus_on_appearing allows clicking to gain focus, just not automatic focus
        imgui.begin(
            "##SliderTooltip",
            flags=(
                imgui.WindowFlags_.no_title_bar |
                imgui.WindowFlags_.no_move |
                imgui.WindowFlags_.no_resize |
                imgui.WindowFlags_.always_auto_resize |
                imgui.WindowFlags_.no_focus_on_appearing |
                imgui.WindowFlags_.no_nav
            )
        )

        # Display the shader-rendered tooltip graphic
        imgui.image(
            self.tooltip_texture_id,
            imgui.ImVec2(self.tooltip_texture_size, self.tooltip_texture_size)
        )

        # Display slider name and description
        imgui.separator()
        imgui.text(f"Parameter: {self.last_hovered_slider}")
        imgui.separator()
        imgui.text_wrapped(self.last_hovered_description)

        # Check if tooltip itself is hovered (must be after content is rendered)
        tooltip_hovered = imgui.is_window_hovered()

        imgui.end()

        # Now decide if we should keep the tooltip visible next frame
        # Keep it if: physics window hovered, tooltip hovered, or actively dragging
        if not physics_window_hovered and not tooltip_hovered and not self.physics_window_interaction:
            self.last_hovered_slider = None

        # Reset interaction flag for next frame
        self.physics_window_interaction = False

    def _refresh_config_files(self):
        """Scan physics_configs directory for .json files, organized by category."""
        self.config_files = []  # Keep for backward compatibility
        self.config_files_by_category = {
            "Core": [],
            "Custom": [],
            "Advanced": []
        }

        if self.configs_dir.exists():
            # Scan Core subfolder
            core_dir = self.configs_dir / "Core"
            if core_dir.exists():
                for f in sorted(core_dir.glob("*.json")):
                    self.config_files_by_category["Core"].append(f.stem)

            # Scan Custom (root level configs, not in subfolders)
            for f in sorted(self.configs_dir.glob("*.json")):
                self.config_files_by_category["Custom"].append(f.stem)
                self.config_files.append(f.stem)  # Maintain backward compat list

            # Scan Advanced subfolder
            advanced_dir = self.configs_dir / "Advanced"
            if advanced_dir.exists():
                for f in sorted(advanced_dir.glob("*.json")):
                    self.config_files_by_category["Advanced"].append(f.stem)

    def _cache_all_configs(self):
        """Load and cache all config files for preview."""
        self._refresh_config_files()
        self.cached_configs = {}

        # Load Core configs from Core subfolder
        for filename in self.config_files_by_category["Core"]:
            filepath = self.configs_dir / "Core" / f"{filename}.json"
            config = self.config_saver.load_from_file(filepath)
            if config:
                self.cached_configs[filename] = config

        # Load Custom configs from root directory
        for filename in self.config_files_by_category["Custom"]:
            filepath = self.configs_dir / f"{filename}.json"
            config = self.config_saver.load_from_file(filepath)
            if config:
                self.cached_configs[filename] = config

        # Load Advanced configs from Advanced subfolder
        for filename in self.config_files_by_category["Advanced"]:
            filepath = self.configs_dir / "Advanced" / f"{filename}.json"
            config = self.config_saver.load_from_file(filepath)
            if config:
                self.cached_configs[filename] = config

    def _get_config_path(self, filename: str) -> Path:
        """Get the full path to a config file by searching all categories.

        Args:
            filename: Config filename without extension

        Returns:
            Path to the config file
        """
        # Check Core folder
        if filename in self.config_files_by_category.get("Core", []):
            return self.configs_dir / "Core" / f"{filename}.json"

        # Check Advanced folder
        if filename in self.config_files_by_category.get("Advanced", []):
            return self.configs_dir / "Advanced" / f"{filename}.json"

        # Default to Custom (root directory)
        return self.configs_dir / f"{filename}.json"

    def _render_load_submenu_content(self, menu_watercolor_mode: bool) -> str | None:
        """Render the content of a load submenu with hierarchical categories.

        Args:
            menu_watercolor_mode: The watercolor mode for this menu (False=standard, True=watercolor)

        Returns:
            The hovered filename this frame, or None
        """
        # Check if we have any configs at all
        total_configs = sum(len(files) for files in self.config_files_by_category.values())
        if total_configs == 0:
            imgui.text_colored(imgui.ImVec4(1.0, 0.5, 0.5, 1.0), "No config files")
            return None

        # Calculate max filename width across all categories
        max_text_width = 0.0
        for category_files in self.config_files_by_category.values():
            for fn in category_files:
                text_size = imgui.calc_text_size(fn)
                if text_size.x > max_text_width:
                    max_text_width = text_size.x

        hovered_this_frame = None

        # Render each category with collapsible headers
        categories = [
            ("Core", self.config_files_by_category["Core"], "load_menu_core_open"),
            ("Custom", self.config_files_by_category["Custom"], "load_menu_custom_open"),
            ("Advanced", self.config_files_by_category["Advanced"], "load_menu_advanced_open")
        ]

        for category_name, category_files, pref_attr in categories:
            if len(category_files) == 0:
                continue  # Skip empty categories

            # Set collapse state from preferences
            imgui.set_next_item_open(getattr(self.state.preferences, pref_attr))
            category_open = imgui.collapsing_header(category_name)

            # Update preference to match actual header state (handles user clicks)
            if imgui.is_item_toggled_open():
                setattr(self.state.preferences, pref_attr, category_open)

            if category_open:
                # Render configs in this category
                for filename in category_files:
                    # Selectable for filename with calculated width
                    clicked, _ = imgui.selectable(
                        filename, False,
                        imgui.SelectableFlags_.no_auto_close_popups,
                        imgui.ImVec2(max_text_width + 10, 0)
                    )

                    # Check if filename is hovered
                    if imgui.is_item_hovered():
                        hovered_this_frame = filename

                    # X button on same line (right after the selectable)
                    imgui.same_line()
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                    if imgui.small_button(f"X##{category_name}_{filename}"):
                        self.delete_confirm_filename = filename
                    imgui.pop_style_color(2)

                    # Also check hover on X button for preview
                    if imgui.is_item_hovered():
                        hovered_this_frame = filename

                    if clicked:
                        # Multi-load mode: add to service directly without closing menu
                        if self.state.multi_load.multi_load_enabled:
                            # Get config from cache or load it
                            if filename in self.cached_configs:
                                config = self.cached_configs[filename]
                                if self.multi_load_service:
                                    success = self.multi_load_service.add_config(config, filename)
                                    if success:
                                        print(f"Config added to multi-load: {filename}")
                                    else:
                                        print(f"Failed to add config: multi-load list is full ({self.multi_load_service.get_config_count()}/64)")
                        # Normal mode: finalize selection (closes menu)
                        else:
                            self._load_filename = filename
                            self._request_load_file = True
                            self._load_watercolor_override = menu_watercolor_mode
                            self.currently_open_project = filename
                            # Clear everything to prevent hover code from re-applying
                            self.cached_config = None
                            self.cached_configs = {}
                            self.currently_previewing = None
                            self.preview_rule_pushed = False
                            imgui.close_current_popup()

        return hovered_this_frame

    def _generate_rule_label(self) -> tuple[int, str, str]:
        """Generate random jersey number with colored digits.

        Returns:
            tuple: (jersey_number, digit1_rgb_string, digit2_rgb_string)
                   e.g., (42, "255,128,64", "64,255,128")
        """
        import colorsys

        # Generate random jersey number (00-99)
        jersey_number = random.randint(0, 99)

        label_digits = []
        for _ in range(2):
            hue = random.randint(0, 255) / 255.0
            sat = random.randint(0, 150) / 255.0
            val = 1.0  # Brightness fixed at 255

            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            r_int, g_int, b_int = int(r * 255), int(g * 255), int(b * 255)
            label_digits.append(f"{r_int},{g_int},{b_int}")

        return (jersey_number, label_digits[0], label_digits[1])

    def _parse_rgb_color(self, rgb_string: str) -> tuple[float, float, float, float]:
        """Parse RGB string to ImVec4 color."""
        r, g, b = map(int, rgb_string.split(','))
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)

    def slider_float_with_range_menu(self, label, param_name, value, default_min, default_max, format="%.3f"):
        """
        Create a slider with an adjustable min/max context menu and reset to defaults.
        Right-click the slider to adjust its range or reset value.

        Args:
            label: Display label for the slider
            param_name: Parameter name (key in current_physics_defaults.values)
            value: Current value
            default_min: Default minimum value
            default_max: Default maximum value
            format: Display format string

        Returns:
            tuple: (changed, new_value)
        """
        # Initialize or get current range
        if label not in self.state.sim.slider_ranges:
            self.state.sim.slider_ranges[label] = [default_min, default_max, default_min, default_max]

        min_val, max_val = self.state.sim.slider_ranges[label][0], self.state.sim.slider_ranges[label][1]

        # Create the slider
        changed, new_value = imgui.slider_float(label, value, min_val, max_val, format=format)

        # Add context menu
        _, _, reset_requested, _ = self.add_slider_context_menu(label, default_min, default_max)

        # If reset was requested, get the default value from current_physics_defaults
        if reset_requested:
            new_value = self.current_physics_defaults.values.get(param_name, value)
            changed = True

        return changed, new_value

    def add_slider_context_menu(self, slider_name, default_min, default_max):
        """
        Add a right-click context menu to adjust slider min/max values and reset to defaults.
        Call this immediately after imgui.slider_float().

        Args:
            slider_name: Unique identifier for this slider
            default_min: Default minimum value
            default_max: Default maximum value

        Returns:
            tuple: (current_min, current_max, reset_requested, range_changed)
        """
        # Initialize slider range if not exists
        if slider_name not in self.state.sim.slider_ranges:
            self.state.sim.slider_ranges[slider_name] = [default_min, default_max, default_min, default_max]

        min_val, max_val, def_min, def_max = self.state.sim.slider_ranges[slider_name]
        range_changed = False
        reset_requested = False

        # Create context menu (right-click on the previous item)
        if imgui.begin_popup_context_item(f"{slider_name}_context"):
            imgui.text(f"Adjust Range: {slider_name}")
            imgui.separator()

            # Min/Max input fields
            changed_min, new_min = imgui.input_float(f"Min##{slider_name}", min_val)
            changed_max, new_max = imgui.input_float(f"Max##{slider_name}", max_val)

            if changed_min:
                self.state.sim.slider_ranges[slider_name][0] = new_min
                range_changed = True
            if changed_max:
                self.state.sim.slider_ranges[slider_name][1] = new_max
                range_changed = True

            imgui.separator()

            # Reset range to default button
            if imgui.button(f"Reset Range to Default##{slider_name}"):
                self.state.sim.slider_ranges[slider_name][0] = def_min
                self.state.sim.slider_ranges[slider_name][1] = def_max
                range_changed = True

            imgui.separator()

            # Reset value button (uses current_physics_defaults)
            if self.current_physics_defaults.source_filename:
                button_label = f"Reset value to '{self.current_physics_defaults.source_filename}'##{slider_name}"
            else:
                button_label = f"Reset value to defaults##{slider_name}"

            if imgui.button(button_label):
                reset_requested = True

            imgui.end_popup()

        return self.state.sim.slider_ranges[slider_name][0], self.state.sim.slider_ranges[slider_name][1], reset_requested, range_changed

    def render_aligned_label(self, label_text: str):
        """Render a right-justified label aligned to the longest label width for consistent button positioning.

        Args:
            label_text: The label text to display (e.g., "Axial Force:")
        """
        # Calculate the width of the longest label to ensure alignment
        longest_label = "Global Force Mult:"
        longest_width = imgui.calc_text_size(longest_label).x
        current_width = imgui.calc_text_size(label_text).x

        # Calculate where to start the label so it ends at the same X position (right-justified)
        label_start_x = imgui.get_style().window_padding.x + longest_width - current_width

        # Position cursor for right-justified label
        imgui.set_cursor_pos_x(label_start_x)
        imgui.text(label_text)
        imgui.same_line()

        # Position cursor at consistent X location for buttons
        target_x = imgui.get_style().window_padding.x + longest_width + 8
        imgui.set_cursor_pos_x(target_x)

    def _set_exclusive_sweep(self, axis: str, param_name: str, new_mode: float):
        """Set a sweep mode, clearing any other active sweep on the same axis.

        Enforces single-sweep-per-axis: only one parameter can be swept on X, Y, or Cohort.

        Args:
            axis: 'x', 'y', or 'cohort'
            param_name: Name of the parameter to set
            new_mode: New sweep mode (0.0 = off, 1.0 = normal, -1.0 = inverse)
        """
        if axis == 'x':
            sweeps = self.state.sim.x_sweeps
        elif axis == 'y':
            sweeps = self.state.sim.y_sweeps
        else:  # cohort
            sweeps = self.state.sim.cohort_sweeps

        # If turning on a sweep, clear all others on this axis first
        if new_mode != 0.0:
            for key in sweeps:
                if key != param_name:
                    sweeps[key] = 0.0

        sweeps[param_name] = new_mode

    def render_sweep_buttons(self, param_name: str):
        """Render X, Y, C sweep toggle buttons for a parameter.

        Left-click cycles: off -> normal -> off
        Right-click cycles: off -> inverse -> off

        Sweep modes: 0.0 = off, 1.0 = normal (highlight), -1.0 = inverse (lowlight)

        Args:
            param_name: Name of the parameter (e.g., 'AXIAL_FORCE')
        """
        button_height = imgui.get_frame_height() * 1.75  # Slightly taller to give range buttons more room
        button_width = button_height * 1.  # Wider than tall

        # X button (Red)
        x_mode = self.state.sim.x_sweeps.get(param_name, 0.0)
        if x_mode == 1.0:  # Normal sweep - bright red (highlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.6, 0.15, 0.15, 1.0))
        elif x_mode == -1.0:  # Inverse sweep - dark red (lowlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3*.3, 0.05*.3, 0.05*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.4*.3, 0.1*.3, 0.1*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.2, 0.03, 0.03, 1.0))
        else:  # Off - dim red
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.4, 0.1, 0.1, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.6, 0.15, 0.15, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.3, 0.08, 0.08, 1.0))

        imgui.button(f"X##{param_name}_x", imgui.ImVec2(button_width, button_height))
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            new_mode = 1.0 if x_mode == 0.0 else 0.0
            self._set_exclusive_sweep('x', param_name, new_mode)
        elif imgui.is_item_clicked(imgui.MouseButton_.right):
            new_mode = -1.0 if x_mode == 0.0 else 0.0
            self._set_exclusive_sweep('x', param_name, new_mode)

        imgui.pop_style_color(3)
        imgui.same_line(spacing=2)

        # Y button (Green)
        y_mode = self.state.sim.y_sweeps.get(param_name, 0.0)
        if y_mode == 1.0:  # Normal sweep - bright green (highlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.8, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 1.0, 0.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.6, 0.15, 1.0))
        elif y_mode == -1.0:  # Inverse sweep - dark green (lowlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.05*.3, 0.3*.3, 0.05*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.1*.3, 0.4*.3, 0.1*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.03, 0.2, 0.03, 1.0))
        else:  # Off - dim green
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.1, 0.4, 0.1, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.15, 0.6, 0.15, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.08, 0.3, 0.08, 1.0))

        imgui.button(f"Y##{param_name}_y", imgui.ImVec2(button_width, button_height))
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            new_mode = 1.0 if y_mode == 0.0 else 0.0
            self._set_exclusive_sweep('y', param_name, new_mode)
        elif imgui.is_item_clicked(imgui.MouseButton_.right):
            new_mode = -1.0 if y_mode == 0.0 else 0.0
            self._set_exclusive_sweep('y', param_name, new_mode)

        imgui.pop_style_color(3)
        imgui.same_line(spacing=2)

        # C button (Yellow)
        c_mode = self.state.sim.cohort_sweeps.get(param_name, 0.0)
        if c_mode == 1.0:  # Normal sweep - bright yellow (highlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.9, 0.9, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 1.0, 0.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.7, 0.7, 0.15, 1.0))
        elif c_mode == -1.0:  # Inverse sweep - dark yellow/brown (lowlight)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3*.3, 0.3*.3, 0.05*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.4*.3, 0.4*.3, 0.1*.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.2*.3, 0.2*.3, 0.03*.3, 1.0))
        else:  # Off - dim yellow
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.4, 0.4, 0.1, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.6, 0.6, 0.15, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.3, 0.3, 0.08, 1.0))

        imgui.button(f"C##{param_name}_c", imgui.ImVec2(button_width, button_height))
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            new_mode = 1.0 if c_mode == 0.0 else 0.0
            self._set_exclusive_sweep('cohort', param_name, new_mode)
        elif imgui.is_item_clicked(imgui.MouseButton_.right):
            new_mode = -1.0 if c_mode == 0.0 else 0.0
            self._set_exclusive_sweep('cohort', param_name, new_mode)

        imgui.pop_style_color(3)

    def adjust_slider_range(self, slider_label: str, current_value: float, default_min: float, default_max: float, widen: bool, strength: float = 2.0, hard_min: float = None, hard_max: float = None):
        """Adjust the min/max range for a slider, widening or narrowing around current value.

        Args:
            slider_label: Label of the slider (e.g., 'Axial Force')
            current_value: Current slider value (x in the formula)
            default_min: Default minimum value
            default_max: Default maximum value
            widen: True to widen, False to narrow
            strength: Strength parameter (S in formula when widening, 1/S when narrowing)
            hard_min: Optional hard minimum limit (e.g., -1.0 for Drag/Sensor Angle)
            hard_max: Optional hard maximum limit (e.g., 1.0 for Drag/Sensor Angle)
        """
        # Get current range - handle both 2-element and 4-element formats
        if slider_label in self.state.sim.slider_ranges:
            range_data = self.state.sim.slider_ranges[slider_label]
            # slider_ranges can be [L, H] or [L, H, default_min, default_max]
            L, H = range_data[0], range_data[1]
        else:
            L, H = default_min, default_max

        # Calculate S based on widen/narrow
        x = current_value
        S = strength if widen else (1.0 / strength)

        # Apply the formulas
        # L' = x - S*( (x-L)/2. + (H-L)/4. )
        # H' = L' + S*(H-L)
        L_prime = x - S * ((x - L) / 2.0 + (H - L) / 4.0)
        H_prime = L_prime + S * (H - L)

        # Apply hard limits if specified (for sliders like Drag and Sensor Angle)
        if hard_min is not None:
            L_prime = max(L_prime, hard_min)
        if hard_max is not None:
            H_prime = min(H_prime, hard_max)

        # Update the range in preferences
        self.state.sim.slider_ranges[slider_label] = [L_prime, H_prime,default_min,default_max]

    def render_range_adjust_buttons(self, param_name: str, slider_label: str, current_value: float, default_min: float, default_max: float, hard_min: float = None, hard_max: float = None):
        """Render widen/narrow buttons for adjusting slider range.

        Args:
            param_name: Name of the parameter (e.g., 'AXIAL_FORCE')
            slider_label: Label of the slider (e.g., 'Axial Force')
            current_value: Current slider value
            default_min: Default minimum value
            default_max: Default maximum value
            hard_min: Optional hard minimum limit (e.g., -1.0 for Drag/Sensor Angle)
            hard_max: Optional hard maximum limit (e.g., 1.0 for Drag/Sensor Angle)
        """
        # Match the height of the XYC sweep buttons (which are 1.35x frame height)
        total_height = imgui.get_frame_height() * 1.43
        button_height = total_height / 2.0  # Half height for stacked buttons
        button_width = imgui.get_frame_height() * 1.23  # Same width as sweep buttons

        # Begin a group to keep buttons together
        imgui.begin_group()
        imgui.push_font(self.default_font,12)
        # Widen button
        if imgui.button(f"^##widen_{param_name}", imgui.ImVec2(button_width, button_height)):
            self.adjust_slider_range(slider_label, current_value, default_min, default_max, widen=True, hard_min=hard_min, hard_max=hard_max)

        # Narrow button
        if imgui.button(f"v##narrow_{param_name}", imgui.ImVec2(button_width, button_height)):
            self.adjust_slider_range(slider_label, current_value, default_min, default_max, widen=False, hard_min=hard_min, hard_max=hard_max)
        imgui.pop_font()
        imgui.end_group()

        # Add tooltip when hovering over the button group
        self._delayed_tooltip("Up arrow widens slider range. Down arrow narrows range")

    def cleanup(self):
        self.tooltip_fbo.release()
        self.tooltip_texture.release()
        self.tooltip_program.release()
        self.imgui_renderer.shutdown()
