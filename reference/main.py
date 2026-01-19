import glfw
import moderngl
import time
import numpy as np
from pathlib import Path
from camera import Camera
from sim import Sim, SIZE_OF_ENTITY_STRUCT
from ui import UI
from services import RuleManager, EntityPicker, VideoRecorderService, ConfigSaver, ArrowDebugService, MultiLoadService
from utilities.gl_helpers import readback_rule
from state import load_preferences, save_preferences, SimState


class App:
    """Main application with Orchestrator pattern."""

    def __init__(self):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        self.window = glfw.create_window(800, 600, "Particle Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # Initialize ModernGL
        self.ctx = moderngl.create_context()
        self.ctx.gc_mode = 'auto'

        # Always on top
        glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)

        # Load preferences first to get world_size
        loaded_prefs = load_preferences()

        # Create components (no cross-references between UI and sim/camera)
        self.sim = Sim(self.ctx, world_size=loaded_prefs.world_size)
        self.camera = Camera(self.ctx, self.sim, self.window)
        self.ui = UI(self.window, self.ctx, self.sim.view_option_labels)

        # Apply loaded preferences to UI
        self.ui.state.preferences = loaded_prefs
        # Initialize world size tracker with loaded value
        self.ui._last_applied_world_size = loaded_prefs.world_size

        # Create services (Orchestrator owns these)
        self.rule_manager = RuleManager()
        # Divide by 4 to convert from bytes to floats (each float32 is 4 bytes)
        entity_stride = SIZE_OF_ENTITY_STRUCT // 4
        self.entity_picker = EntityPicker(self.sim.get_entity_buffer(), entity_stride)
        self.video_service = VideoRecorderService()
        self.config_saver = ConfigSaver()
        self.arrow_debug_service = ArrowDebugService(self.ctx)
        self.multi_load_service = MultiLoadService()
        self.ui.multi_load_service = self.multi_load_service  # Give UI access to service
        self.configs_dir = Path("physics_configs")
        self.configs_dir.mkdir(exist_ok=True)

        # Preview state
        self.preview_rule_active = False  # File->load preview
        self.history_preview_rule_active = False  # History window preview

        # Frame timing
        self.last_update_time = time.time()

        # Track user's desired speedmult (for restoration after recording)
        self.user_speedmult = 1
        self.was_recording = False

        # Track user's desired motion blur settings (for restoration after recording)
        self.user_motion_blur = True
        self.user_blur_quality = 1

        # Screenshot state machine
        self.screenshot_pending = False  # Waiting for current frame to finish
        self.screenshot_in_progress = False  # Override frame is running
        self.screenshot_saved_settings = {}  # Saved user settings to restore

        # Mouse tracking for draw trail mode
        self.prev_mouse_tex_coords = (0.0, 0.0)
        self.mouse_button_state = False  # Track if left mouse button is currently pressed

        # Track previous view option for camera repositioning when leaving tiling mode
        self.prev_view_option = 0

        # Ensure _Default.json exists and load it
        self._ensure_default_config()
        self._load_default_config()
        self.sim.reload()
        self.sim.reset()

    def _ensure_default_config(self):
        """Ensure _Default.json exists in physics_configs directory. Create it if missing."""
        import numpy as np
        default_path = self.configs_dir / "Core/_Default.json"
        if not default_path.exists():
            # Create default config from fresh SimState
            default_state = SimState()
            # Create zero rule for the default config
            zero_rule = np.zeros((10, 8), dtype=np.float32)
            # Create and save config
            config = self.config_saver.create_config(default_state, zero_rule)
            self.config_saver.save_to_file(config, default_path)
            print(f"Created default config: {default_path}")

    def _load_default_config(self):
        """Load _Default.json on startup."""
        default_path = self.configs_dir / "Core/_Default.json"
        config = self.config_saver.load_from_file(default_path)
        if config is not None:
            rule = self.config_saver.apply_config(config, self.ui.state.sim)
            self.rule_manager.push_rule(rule)
            self.sim.apply_rule(rule)
            print(f"Loaded default config from {default_path}")
            self.ui.update_physics_defaults("_Default")
        else:
            print(f"Failed to load default config from {default_path}")

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.orchestrate_frame()
            glfw.swap_buffers(self.window)

        self.cleanup()

    def orchestrate_frame(self):
        """Main orchestration logic - reads UI state, coordinates components."""

        # 1. Get current UI state
        ui_state = self.ui.get_state()
        
        tiling_mode = (ui_state.sim.current_view_option == 3)
        # 2. Process one-shot commands
        self.process_commands(ui_state,tiling_mode)

        # 3. Process continuous input (camera movement)
        self.process_camera_input(ui_state)

        # 3.5. Screenshot state machine
        # If screenshot_pending was set on previous frame, start the override frame now
        if self.screenshot_pending and not self.screenshot_in_progress:
            # Transition from pending to in_progress
            self.screenshot_pending = False
            self.screenshot_in_progress = True
            # Save current settings
            self.screenshot_saved_settings = {
                'speedmult': ui_state.preferences.speedmult,
                'blur_quality': ui_state.preferences.blur_quality,
                'motion_blur': ui_state.preferences.motion_blur,
                'going': ui_state.sim.going,
            }
            # Apply override settings for maximum quality screenshot
            # speedmult = motion_blur_samples (N physics steps per frame)
            # blur_quality = 1 (render every physics step for N total samples)
            # motion_blur = True (enable temporal accumulation)
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.blur_quality = 1
            ui_state.preferences.motion_blur = True
            # If paused, temporarily unpause for this frame
            if not ui_state.sim.going:
                ui_state.sim.going = True

        # 4. Lock physics frequency to video recorder frequency if recording
        is_recording = self.video_service.is_active()

        # Detect recording state changes
        if is_recording and not self.was_recording:
            # Recording just started - save user's speedmult and motion blur settings
            self.user_speedmult = ui_state.preferences.speedmult
            self.user_motion_blur = ui_state.preferences.motion_blur
            self.user_blur_quality = ui_state.preferences.blur_quality
        elif not is_recording and self.was_recording:
            # Recording just stopped - restore user's speedmult and motion blur settings
            ui_state.preferences.speedmult = self.user_speedmult
            ui_state.preferences.motion_blur = self.user_motion_blur
            ui_state.preferences.blur_quality = self.user_blur_quality

        # Lock speedmult and motion blur settings while recording
        if is_recording:
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.motion_blur = ui_state.preferences.recording_motion_blur
            ui_state.preferences.blur_quality = ui_state.preferences.recording_blur_quality

        # Update recording state for next frame
        self.was_recording = is_recording

        # 5. Apply state to components
        self.sim.apply_state(ui_state.sim)
        self.sim.apply_camera_state(ui_state.camera)
        self.camera.apply_state(ui_state.camera)
        self.multi_load_service.apply_state(ui_state.multi_load)
        # Sync brightness from preferences
        self.camera.BRIGHTNESS = ui_state.preferences.brightness

        # 5.1. Multi-load conflict prevention
        if ui_state.multi_load.multi_load_enabled:
            # Disable parameter sweeps when multi-load is active
            ui_state.sim.parameter_sweeps_enabled = False
            # Force mouse mode to Draw Trail
            ui_state.preferences.mouse_mode = "Draw Trail"

        # 5.5. Calculate sweep reticle info (needed for both running and paused states)
        sweep_reticle_x, sweep_reticle_y, sweep_reticle_visible = self.sim.get_sweep_reticle_position()

        # Hide reticle when recording video
        if is_recording or self.screenshot_in_progress:
            sweep_reticle_visible = False

        # Transform reticle from texture UV to screen UV (accounting for camera)
        width, height = glfw.get_framebuffer_size(self.window)
        screen_aspect = width / height if height > 0 else 1.0

        if sweep_reticle_visible:
            # Convert texture coords to screen pixels
            screen_x, screen_y = self.camera.tex_to_screen(
                (sweep_reticle_x, sweep_reticle_y),
                self.sim.view_tex.size
            )
            # Convert screen pixels to screen UV (0-1)
            sweep_reticle_x = screen_x / width
            sweep_reticle_y = screen_y / height

        sweep_mode = ui_state.sim.parameter_sweeps_enabled
        sweep_reticle_pos = (sweep_reticle_x, sweep_reticle_y)



        # Reposition camera when leaving tiling mode to keep it over the fundamental period
        if self.prev_view_option == 3 and ui_state.sim.current_view_option != 3:
            # We just left tiling mode - wrap camera position to [0, 2) using modular arithmetic
            # Camera position is in world space where the canvas spans [-1, 1]
            # The fundamental period is 2.0 (from -1 to 1)
            ui_state.camera.position[0] = np.fmod(ui_state.camera.position[0] + 100.0, 2.0) - 1.0
            ui_state.camera.position[1] = np.fmod(ui_state.camera.position[1] + 100.0, 2.0) - 1.0

        # Update previous view option for next frame
        self.prev_view_option = ui_state.sim.current_view_option

        # 6. Run simulation if going
        if ui_state.sim.going:
            self.run_simulation_frame(ui_state, sweep_mode, sweep_reticle_pos, sweep_reticle_visible,
                                      screen_aspect, ui_state.sim.watercolor_mode, tiling_mode=tiling_mode)

        # 6.5. Screenshot save and settings restoration
        if self.screenshot_in_progress:
            # Save the screenshot using the assembled texture
            if self.camera.assembled_texture is not None:
                from utilities.save_frame_gpu import save_frame_gpu
                import datetime
                # Generate timestamped filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = ui_state.preferences.filename_prefix or "screenshot"
                filename = save_frame_gpu(
                    self.camera.assembled_texture,
                    self.ctx,
                    supersample_k=ui_state.preferences.supersample_k,
                    return_array=False
                )
                # Rename to proper filename with timestamp
                import os
                if filename and os.path.exists(filename):
                    new_filename = f"Screenshots/{prefix}_{timestamp}.png"
                    os.rename(filename, new_filename)
                    print(f"Screenshot saved: {new_filename}")

            # Restore saved settings
            ui_state.preferences.speedmult = self.screenshot_saved_settings['speedmult']
            ui_state.preferences.blur_quality = self.screenshot_saved_settings['blur_quality']
            ui_state.preferences.motion_blur = self.screenshot_saved_settings['motion_blur']
            ui_state.sim.going = self.screenshot_saved_settings['going']
            self.screenshot_in_progress = False
            self.screenshot_saved_settings = {}

        # 7. Render camera view
        # Determine emboss texture based on mode: 0=Off (None), 1=Canvas, 2=Brush
        emboss_mode = ui_state.sim.emboss_mode
        if emboss_mode == 1:
            emboss_tex = self.sim.can
        elif emboss_mode == 2:
            emboss_tex = self.sim.brush_tex
        else:
            emboss_tex = None
        # Check if in draw trail mode
        draw_trail_mode = ui_state.preferences.mouse_mode == "Draw Trail"

        # Convert mouse position to normalized screen coords (0-1 range)
        width, height = glfw.get_framebuffer_size(self.window)
        mouse_x_norm = ui_state.mouse_pos[0] / width if width > 0 else 0.5
        mouse_y_norm = ui_state.mouse_pos[1] / height if height > 0 else 0.5
        mouse_screen_coords = (mouse_x_norm, mouse_y_norm)

        self.camera.render(
            sim_going=ui_state.sim.going,
            current_view_option=ui_state.sim.current_view_option,
            sweep_mode=sweep_mode,
            sweep_reticle_pos=sweep_reticle_pos,
            sweep_reticle_visible=sweep_reticle_visible,
            screen_aspect=screen_aspect,
            watercolor_mode=ui_state.sim.watercolor_mode,
            ink_weight=ui_state.sim.ink_weight,
            emboss_tex=emboss_tex,
            emboss_mode=emboss_mode,
            emboss_intensity=ui_state.sim.emboss_intensity,
            emboss_smoothness=ui_state.sim.emboss_smoothness,
            draw_trail_mode=draw_trail_mode,
            draw_size=ui_state.preferences.draw_size,
            mouse_screen_coords=mouse_screen_coords,
            exposure=ui_state.preferences.exposure,
            tiling_mode=tiling_mode
        )

        # 7.5. Render arrow debug overlay if enabled
        if ui_state.preferences.debug_arrows:
            width, height = glfw.get_framebuffer_size(self.window)
            self.arrow_debug_service.render(
                canvas_texture=self.sim.can,
                cam_pos=tuple(self.camera.position),
                cam_zoom=self.camera.zoom,
                canvas_resolution=self.sim.can.size,
                window_size=(width, height),
                arrow_sensitivity=ui_state.preferences.arrow_sensitivity
            )

        # 8. Update UI display info and render
        self.ui.update_display_info({
            'time': self.sim.time,
            'frame_count': self.sim.frame_count,
            'tex_size': self.sim.view_tex.size,
            'recording_active': self.video_service.is_active(),
            'rule_history': self.rule_manager.rule_history,
        })
        self.ui.render()

    def process_commands(self, ui_state,tiling_mode):
        """Handle one-shot commands."""

        # Handle world size change
        if ui_state.request_world_size_change:
            # Update sim's world_size
            self.sim.world_size = ui_state.preferences.world_size
            # Reallocate buffers and textures
            self.sim.setup_simulation_state()
            # Recompile shaders with new entity count
            self.sim.setup_shaders()
            # Update entity picker with new buffer
            self.entity_picker.update_buffer(self.sim.get_entity_buffer())
            # Apply current rule
            if self.rule_manager.has_rules():
                self.sim.apply_rule(self.rule_manager.get_current_rule())
            # Reset simulation
            self.sim.reset()
            # Update UI tracker so we don't trigger again
            self.ui._last_applied_world_size = ui_state.preferences.world_size
            print(f"World size changed to {self.sim.world_size} (entity_count: {self.sim.entity_count}, canvas: {self.sim.get_canvas_dimensions()}x{self.sim.get_canvas_dimensions()})")

        # Toggle recording
        if ui_state.toggle_recording:
            self.video_service.toggle()

        # Screenshot request (Shift+P) - set pending flag
        if ui_state.request_screenshot and not self.screenshot_pending and not self.screenshot_in_progress:
            self.screenshot_pending = True

        # Shader reload
        if ui_state.request_reload:
            self.sim.reload()
            if self.rule_manager.has_rules():
                self.sim.apply_rule(self.rule_manager.get_current_rule())
            self.camera.reload()

        # Simple reset (R key)
        if ui_state.request_reset:
            self.sim.reset()

        # Full reset (Z key) - push zero rule (undoable) and reset entities
        if ui_state.request_full_reset:
            self.sim.reset()
            zero_rule = self.rule_manager.push_zero_rule()
            self.sim.apply_rule(zero_rule)

        # Handle sweep preview restore: ANY click (including on imgui) re-enables sweeps
        restored_sweep_preview = False
        if ui_state.sim.sweep_preview_pending_restore:
            if ui_state.any_left_click_this_frame or ui_state.any_right_click_this_frame:
                ui_state.sim.parameter_sweeps_enabled = True
                ui_state.sim.sweep_preview_pending_restore = False
                restored_sweep_preview = True  # Skip click handling below

        # Handle mouse clicks - behavior depends on whether parameter sweeps are enabled
        if not restored_sweep_preview:
            if ui_state.left_click_this_frame:
                # When parameter sweeps are enabled, left click updates sliders (any mouse mode)
                if ui_state.sim.parameter_sweeps_enabled:
                    # Only update sliders if there are active sweeps
                    if self.sim.has_active_xy_sweep():
                        tex_coords = self.camera.screen_to_tex(
                            ui_state.mouse_pos,
                            self.sim.view_tex.size
                        )
                        # In tiling mode, wrap mouse position to fundamental domain
                        if tiling_mode:
                            tex_coords = (
                                np.fmod(tex_coords[0] + 10.0, 1.0),
                                np.fmod(tex_coords[1] + 10.0, 1.0)
                            )
                        world_pos = (tex_coords[0] * 2 - 1, tex_coords[1] * 2 - 1)

                        if self.sim.has_active_cohort_sweep():
                            entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(tex_coords)
                            self.sim.update_sliders_from_particle(world_pos, entity_cohort)
                        else:
                            self.sim.update_sliders_from_position(world_pos)
                    # If no active sweeps, left click does nothing
                # When parameter sweeps are disabled and in Select Particle mode, pick entity
                elif ui_state.preferences.mouse_mode == "Select Particle":
                    # Normal mode: pick entity and apply rule
                    tex_coords = self.camera.screen_to_tex(
                        ui_state.mouse_pos,
                        self.sim.view_tex.size
                    )
                    # In tiling mode, wrap mouse position to fundamental domain
                    if tiling_mode:
                        tex_coords = (
                            np.fmod(tex_coords[0] + 10.0, 1.0),
                            np.fmod(tex_coords[1] + 10.0, 1.0)
                        )
                    entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(tex_coords)

                    # Bounds check: ensure entity_id is valid for current buffer size
                    if entity_id >= 0 and entity_id < self.sim.entity_count:
                        print(f"Entity {entity_id} at pos {entity_pos}, cohort {entity_cohort}")
                        rule = readback_rule(self.sim.get_rule_buffer(), entity_id)
                        self.rule_manager.push_rule(rule)
                        self.sim.apply_rule(rule)
                        self.sim.update_sliders_from_particle(entity_pos, entity_cohort)
                    else:
                        print(f"Warning: entity_id {entity_id} out of bounds (max: {self.sim.entity_count - 1})")
            elif ui_state.right_click_this_frame:
                # When parameter sweeps are enabled, right click enters preview mode (any mouse mode)
                if ui_state.sim.parameter_sweeps_enabled:
                    # Only enter preview mode if there are active sweeps
                    if self.sim.has_active_xy_sweep():
                        # Enter sweep preview mode: disable sweeps and set pending restore
                        ui_state.sim.parameter_sweeps_enabled = False
                        ui_state.sim.sweep_preview_pending_restore = True
                    # If no active sweeps, right click does nothing
                # When parameter sweeps are disabled and in Select Particle mode, undo rule
                elif ui_state.preferences.mouse_mode == "Select Particle":
                    # Normal mode: pop rule from history
                    prev_rule = self.rule_manager.pop_rule()
                    self.sim.apply_rule(prev_rule)

        # Handle config save (Ctrl+C)
        if ui_state.request_save_config:
            current_rule = self.rule_manager.get_current_rule()
            config_string = self.config_saver.save_to_string(ui_state.sim, current_rule)
            self.ui.set_clipboard(config_string)
            print(f"Config copied to clipboard ({len(config_string)} chars)")

        # Handle config load (Ctrl+V)
        if ui_state.request_load_config:
            config_string = ui_state.clipboard_text
            if config_string:
                rule = self.config_saver.load_from_string(config_string, ui_state.sim)
                if rule is not None:
                    self.rule_manager.push_rule(rule)
                    self.sim.apply_rule(rule)
                    print("Config loaded from clipboard")
                else:
                    print("Failed to load config from clipboard")

        # Handle file save (menu)
        if ui_state.request_save_file:
            filename = ui_state.save_filename
            if filename:
                current_rule = self.rule_manager.get_current_rule()
                config = self.config_saver.create_config(ui_state.sim, current_rule)
                filepath = self.configs_dir / f"{filename}.json"
                self.config_saver.save_to_file(config, filepath)
                print(f"Config saved to {filepath}")
                self.ui.update_physics_defaults(filename)

        # Handle file load (menu)
        if ui_state.request_load_file:
            filename = ui_state.load_filename
            if filename:
                # Multi-load mode: add config to service instead of replacing current
                if ui_state.multi_load.multi_load_enabled:
                    filepath = self.ui._get_config_path(filename)
                    config = self.config_saver.load_from_file(filepath)
                    if config is not None:
                        success = self.multi_load_service.add_config(config, filename)
                        if success:
                            print(f"Config added to multi-load: {filename}")
                        else:
                            print(f"Failed to add config: multi-load list is full ({self.multi_load_service.get_config_count()}/64)")
                    else:
                        print(f"Failed to load config from {filepath}")
                # Normal mode: load and apply config
                else:
                    if self.preview_rule_active:
                        # Preview already applied config and pushed rule - just finalize it
                        self.preview_rule_active = False
                        # Apply watercolor override if provided
                        if ui_state.load_watercolor_override is not None:
                            ui_state.sim.watercolor_mode = ui_state.load_watercolor_override
                        print(f"Config loaded (from preview): {filename}")
                        self.ui.update_physics_defaults(filename)
                    else:
                        # No preview active - load fresh from file
                        filepath = self.ui._get_config_path(filename)
                        config = self.config_saver.load_from_file(filepath)
                        if config is not None:
                            rule = self.config_saver.apply_config(
                                config, ui_state.sim,
                                watercolor_override=ui_state.load_watercolor_override
                            )
                            self.rule_manager.push_rule(rule)
                            self.sim.apply_rule(rule)
                            print(f"Config loaded from {filepath}")
                            self.ui.update_physics_defaults(filename)
                        else:
                            print(f"Failed to load config from {filepath}")

        # Handle file delete (menu)
        if ui_state.request_delete_file:
            filename = ui_state.delete_filename
            if filename:
                filepath = self.ui._get_config_path(filename)
                if filepath.exists():
                    filepath.unlink()
                    print(f"Config deleted: {filepath}")

        # Handle clear preview (unhover or close submenu) - must happen before new preview
        if ui_state.request_clear_preview:
            if self.preview_rule_active:
                prev_rule = self.rule_manager.pop_rule()
                self.sim.apply_rule(prev_rule)
                self.preview_rule_active = False

        # Handle config preview (hover in Load submenu)
        if ui_state.request_preview_config:
            filename = ui_state.preview_filename
            if filename:
                filepath = self.ui._get_config_path(filename)#self.configs_dir / f"{filename}.json"
                config = self.config_saver.load_from_file(filepath)
                if config and config.rule is not None:
                    self.rule_manager.push_rule(config.rule)
                    self.sim.apply_rule(config.rule)
                    self.preview_rule_active = True

        # Handle rule history preview - clear must happen BEFORE new preview
        if ui_state.request_clear_history_preview:
            if self.history_preview_rule_active:
                prev_rule = self.rule_manager.pop_rule()
                self.ui.history_window_labels.pop()  # Also remove the preview's label
                self.sim.apply_rule(prev_rule)
                self.history_preview_rule_active = False

        if ui_state.request_preview_history_rule:
            idx = ui_state.history_preview_index
            if 0 <= idx < len(self.rule_manager.rule_history):
                rule_to_preview = self.rule_manager.rule_history[idx].copy()
                self.rule_manager.push_rule(rule_to_preview)
                self.sim.apply_rule(rule_to_preview)
                self.history_preview_rule_active = True

        if ui_state.request_load_history_rule:
            # Clear preview first
            if self.history_preview_rule_active:
                self.rule_manager.pop_rule()
                self.ui.history_window_labels.pop()  # Also remove the preview's label
                self.history_preview_rule_active = False

            idx = ui_state.history_preview_index
            if 0 <= idx < len(self.rule_manager.rule_history):
                # Move rule to top with metadata
                rule_to_load = self.rule_manager.rule_history[idx].copy()
                label_to_preserve = self.ui.history_window_labels[idx]

                self.rule_manager.rule_history.pop(idx)
                self.ui.history_window_labels.pop(idx)

                self.rule_manager.push_rule(rule_to_load)
                self.ui.history_window_labels.append(label_to_preserve)
                self.sim.apply_rule(rule_to_load)

        if ui_state.request_delete_history_rule:
            # Clear preview first
            if self.history_preview_rule_active:
                self.rule_manager.pop_rule()
                self.ui.history_window_labels.pop()  # Also remove the preview's label
                self.history_preview_rule_active = False

            idx = ui_state.history_preview_index
            if 0 <= idx < len(self.rule_manager.rule_history):
                self.rule_manager.rule_history.pop(idx)
                self.ui.history_window_labels.pop(idx)

                current_rule = self.rule_manager.get_current_rule()
                self.sim.apply_rule(current_rule)

    def process_camera_input(self, ui_state):
        """Handle continuous WASD/QE input for camera and scroll zoom."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        move_speed = 2.0 * dt * ui_state.camera.zoom
        zoom_speed = 2.6 * dt

        keys = ui_state.keys_pressed

        # Get key bindings from UI
        key_w = self.ui.keybindings.get_key("camera_forward")
        key_s = self.ui.keybindings.get_key("camera_backward")
        key_a = self.ui.keybindings.get_key("camera_left")
        key_d = self.ui.keybindings.get_key("camera_right")
        key_e = self.ui.keybindings.get_key("camera_in")
        key_q = self.ui.keybindings.get_key("camera_out")

        if key_w and key_w in keys:
            ui_state.camera.position[1] -= move_speed
        if key_s and key_s in keys:
            ui_state.camera.position[1] += move_speed
        if key_a and key_a in keys:
            ui_state.camera.position[0] -= move_speed
        if key_d and key_d in keys:
            ui_state.camera.position[0] += move_speed

        if key_e and key_e in keys:
            ui_state.camera.zoom *= (1.0 - zoom_speed)
        if key_q and key_q in keys:
            ui_state.camera.zoom *= (1.0 + zoom_speed)

        # Handle scroll zoom (zoom around mouse pointer - "Factorio-style")
        if ui_state.scroll_delta != 0.0:
            # Get window dimensions
            width, height = glfw.get_framebuffer_size(self.window)

            # Convert mouse to NDC
            x_screen, y_screen = ui_state.mouse_pos
            x_ndc = (x_screen / width) * 2 - 1
            y_ndc = (1 - y_screen / height) * 2 - 1

            # Calculate aspect ratios
            tex_size = self.sim.view_tex.size
            tex_aspect = tex_size[0] / tex_size[1]
            window_aspect = width / height

            if tex_aspect > window_aspect:
                scale_x = 1.0
                scale_y = window_aspect / tex_aspect
            else:
                scale_x = tex_aspect / window_aspect
                scale_y = 1.0

            # Get world position under mouse BEFORE zoom
            old_zoom = ui_state.camera.zoom
            old_pos = ui_state.camera.position.copy()

            scale_x_old = scale_x / old_zoom
            scale_y_old = scale_y / old_zoom
            x_ndc_adj = x_ndc + old_pos[0] / old_zoom
            y_ndc_adj = y_ndc - old_pos[1] / old_zoom
            world_x = x_ndc_adj / scale_x_old
            world_y = y_ndc_adj / scale_y_old

            # Apply zoom (scroll up = zoom in = smaller zoom value)
            scroll_zoom_speed = 0.1
            zoom_factor = 1.0 - ui_state.scroll_delta * scroll_zoom_speed
            zoom_factor = max(0.5, min(2.0, zoom_factor))  # Clamp zoom step
            new_zoom = old_zoom * zoom_factor
            ui_state.camera.zoom = new_zoom

            # Calculate where the world point would now appear in NDC
            scale_x_new = scale_x / new_zoom
            scale_y_new = scale_y / new_zoom
            new_x_ndc_adj = world_x * scale_x_new
            new_y_ndc_adj = world_y * scale_y_new

            # Adjust camera position so the world point stays at the same screen position
            # We want: new_x_ndc_adj = x_ndc + new_pos[0] / new_zoom
            # So: new_pos[0] = (new_x_ndc_adj - x_ndc) * new_zoom
            ui_state.camera.position[0] = (new_x_ndc_adj - x_ndc) * new_zoom
            ui_state.camera.position[1] = -(new_y_ndc_adj - y_ndc) * new_zoom

    def run_simulation_frame(self, ui_state, sweep_mode: bool, sweep_reticle_pos: tuple,
                              sweep_reticle_visible: bool, screen_aspect: float,
                              watercolor_mode: bool = False, tiling_mode: bool = False):
        """Run simulation step(s) with frame assembly and video recording."""
        # Set watercolor mode on camera for generate_view_texture()
        self.camera.watercolor_mode = watercolor_mode
        speedmult = ui_state.preferences.speedmult
        motion_blur = ui_state.preferences.motion_blur

        # Calculate mouse screen coordinates for draw overlay
        width, height = glfw.get_framebuffer_size(self.window)
        mouse_x_norm = ui_state.mouse_pos[0] / width if width > 0 else 0.5
        mouse_y_norm = ui_state.mouse_pos[1] / height if height > 0 else 0.5
        mouse_screen_coords = (mouse_x_norm, mouse_y_norm)

        # Compute view bounds for tiling mode
        view_min = (0.0, 0.0)
        view_max = (0.0, 0.0)
        if tiling_mode:
            import numpy as np
            view_min_ndc = np.array([-1.0, -1.0])
            view_max_ndc = np.array([1.0, 1.0])
            view_min = view_min_ndc * self.camera.zoom + self.camera.position * np.array([1.0, -1.0])
            view_max = view_max_ndc * self.camera.zoom + self.camera.position * np.array([1.0, -1.0])
            view_min[0] *= screen_aspect
            view_max[0] *= screen_aspect

        # Calculate draw mode parameters
        # Disable trail drawing when parameter sweeps are active
        draw_mode = (ui_state.preferences.mouse_mode == "Draw Trail" and
                     not ui_state.sim.parameter_sweeps_enabled)
        mouse_tex_coords = (0.0, 0.0)
        draw_power_value = 0.0

        if draw_mode:
            # Convert screen mouse position to texture coordinates (0-1 range)
            mouse_tex_coords = self.camera.screen_to_tex(
                ui_state.mouse_pos,
                self.sim.can.size
            )

            # In tiling mode, wrap mouse position to fundamental domain using modular arithmetic
            if tiling_mode:
                mouse_tex_coords = (
                    np.fmod(mouse_tex_coords[0] + 10.0, 1.0),  # +10 ensures positive before fmod
                    np.fmod(mouse_tex_coords[1] + 10.0, 1.0)
                )

            # Only set draw_power if button is pressed (respects imgui capture)
            if ui_state.mouse_left_held:
                draw_power_value = ui_state.preferences.draw_power

        if motion_blur:
            # Motion blur enabled: temporal accumulation with multiple render calls
            # Run simulation steps and accumulate frames
            motion_blur_render_cadence = ui_state.preferences.blur_quality

            # Calculate how many render samples we'll actually take
            # We render every Nth physics step, so total_render_samples = speedmult / cadence
            render_sample_index = 0

            for step in range(speedmult):
                self.sim.update(
                    self.ctx,
                    draw_mode=draw_mode,
                    mouse_pos=mouse_tex_coords,
                    prev_mouse_pos=self.prev_mouse_tex_coords,
                    draw_size=ui_state.preferences.draw_size,
                    draw_power=draw_power_value,
                    multi_load_service=self.multi_load_service if ui_state.multi_load.multi_load_enabled else None,
                    is_preview_active = self.preview_rule_active,
                    tiling_mode=tiling_mode
                )

                # Only render on frames matching the blur quality cadence
                if step % motion_blur_render_cadence != 0:
                    continue

                # Generate raw view texture (PRE-gamma correction)
                raw_view_tex = self.camera.generate_view_texture(tiling_mode=tiling_mode)

                # Assemble frame (applies gamma correction on final sample)
                # Determine emboss texture based on mode: 0=Off (None), 1=Canvas, 2=Brush
                emboss_mode = ui_state.sim.emboss_mode
                if emboss_mode == 1:
                    emboss_tex = self.sim.can
                elif emboss_mode == 2:
                    emboss_tex = self.sim.brush_tex
                else:
                    emboss_tex = None
                # Override emboss_intensity to 0 when mode is Off (0)
                effective_emboss_intensity = 0.0 if emboss_mode == 0 else ui_state.sim.emboss_intensity

                # Calculate total render samples based on cadence
                total_render_samples = (speedmult + motion_blur_render_cadence - 1) // motion_blur_render_cadence

                assembled_tex = self.camera.frame_assembler.assemble_frame(
                    raw_view_tex,
                    total_samples=total_render_samples,
                    current_sample_index=render_sample_index,
                    view_mode=ui_state.sim.current_view_option,
                    sweep_mode=sweep_mode,
                    sweep_reticle_pos=sweep_reticle_pos,
                    sweep_reticle_visible=sweep_reticle_visible,
                    screen_aspect=screen_aspect,
                    brightness=self.camera.BRIGHTNESS,
                    exposure=ui_state.preferences.exposure,
                    ink_weight=ui_state.sim.ink_weight,
                    watercolor_mode=ui_state.sim.watercolor_mode,
                    emboss_tex=emboss_tex,
                    camera_position=tuple(self.camera.position),
                    camera_zoom=self.camera.zoom,
                    emboss_intensity=effective_emboss_intensity,
                    emboss_smoothness=ui_state.sim.emboss_smoothness,
                    trail_draw_radius= ui_state.preferences.draw_size if ui_state.preferences.mouse_mode== "Draw Trail" and (not self.video_service.is_active()) and (not self.screenshot_in_progress) and (not ui_state.sim.parameter_sweeps_enabled) else 0,
                    mouse_screen_coords=mouse_screen_coords,
                    tiling_mode=tiling_mode,
                    view_min=tuple(view_min),
                    view_max=tuple(view_max)
                )

                # Increment render sample index for next sample
                render_sample_index += 1

                # Only process when accumulation cycle completes
                if assembled_tex is not None:
                    self.camera.assembled_texture = assembled_tex

                    # Send to video recorder if recording
                    if self.video_service.is_active():
                        self.video_service.process_frame(
                            self.camera.ctx,
                            assembled_tex,  # Already gamma-corrected and temporally complete
                            ui_state.preferences.max_frames,
                            ui_state.preferences.supersample_k,
                            ui_state.preferences.filename_prefix
                        )
        else:
            # Motion blur disabled: multiple physics steps, single render call
            # Run all simulation updates
            for step in range(speedmult):
                self.sim.update(
                    self.ctx,
                    draw_mode=draw_mode,
                    mouse_pos=mouse_tex_coords,
                    prev_mouse_pos=self.prev_mouse_tex_coords,
                    draw_size=ui_state.preferences.draw_size,
                    draw_power=draw_power_value,
                    multi_load_service=self.multi_load_service if ui_state.multi_load.multi_load_enabled else None,
                    tiling_mode=tiling_mode
                )

            # Generate view texture only once at the end
            raw_view_tex = self.camera.generate_view_texture(tiling_mode=tiling_mode)

            # Apply gamma correction in single-sample mode (no temporal accumulation)
            # Determine emboss texture based on mode: 0=Off (None), 1=Canvas, 2=Brush
            emboss_mode = ui_state.sim.emboss_mode
            if emboss_mode == 1:
                emboss_tex = self.sim.can
            elif emboss_mode == 2:
                emboss_tex = self.sim.brush_tex
            else:
                emboss_tex = None
            # Override emboss_intensity to 0 when mode is Off (0)
            effective_emboss_intensity = 0.0 if emboss_mode == 0 else ui_state.sim.emboss_intensity
            assembled_tex = self.camera.frame_assembler.assemble_frame(
                raw_view_tex,
                total_samples=1,
                current_sample_index=0,
                view_mode=ui_state.sim.current_view_option,
                sweep_mode=sweep_mode,
                sweep_reticle_pos=sweep_reticle_pos,
                sweep_reticle_visible=sweep_reticle_visible,
                screen_aspect=screen_aspect,
                brightness=self.camera.BRIGHTNESS,
                exposure=ui_state.preferences.exposure,
                ink_weight=ui_state.sim.ink_weight,
                watercolor_mode=ui_state.sim.watercolor_mode,
                emboss_tex=emboss_tex,
                camera_position=tuple(self.camera.position),
                camera_zoom=self.camera.zoom,
                emboss_intensity=effective_emboss_intensity,
                emboss_smoothness=ui_state.sim.emboss_smoothness,
                trail_draw_radius= ui_state.preferences.draw_size if ui_state.preferences.mouse_mode== "Draw Trail" and (not self.video_service.is_active()) and (not self.screenshot_in_progress) and (not ui_state.sim.parameter_sweeps_enabled) else 0,
                mouse_screen_coords=mouse_screen_coords,
                tiling_mode=tiling_mode,
                view_min=tuple(view_min),
                view_max=tuple(view_max)
            )

            self.camera.assembled_texture = assembled_tex

            # Send to video recorder if recording
            if self.video_service.is_active():
                self.video_service.process_frame(
                    self.camera.ctx,
                    assembled_tex,
                    ui_state.preferences.max_frames,
                    ui_state.preferences.supersample_k,
                    ui_state.preferences.filename_prefix
                )

        # Update previous mouse position for next frame
        if draw_mode:
            self.prev_mouse_tex_coords = mouse_tex_coords

    def cleanup(self):
        # Save preferences before cleanup
        ui_state = self.ui.get_state()
        save_preferences(ui_state.preferences)

        self.video_service.cleanup()
        self.ui.cleanup()
        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()
