from dataclasses import dataclass, field
from .sim_state import SimState
from .camera_state import CameraState
from .recording_state import RecordingState
from .preferences_state import PreferencesState
from .multi_load_state import MultiLoadState


@dataclass
class UIState:
    """Aggregate state that UI exposes to Orchestrator each frame."""
    sim: SimState = field(default_factory=SimState)
    camera: CameraState = field(default_factory=CameraState)
    recording: RecordingState = field(default_factory=RecordingState)
    preferences: PreferencesState = field(default_factory=PreferencesState)
    multi_load: MultiLoadState = field(default_factory=MultiLoadState)

    # Input state (updated by callbacks)
    keys_pressed: set = field(default_factory=set)
    mouse_pos: tuple = (0.0, 0.0)

    # One-shot click events (reset after get_state)
    left_click_this_frame: bool = False
    right_click_this_frame: bool = False

    # Any click events (includes clicks on imgui elements, for sweep restore)
    any_left_click_this_frame: bool = False
    any_right_click_this_frame: bool = False

    # Continuous mouse state (respects imgui capture)
    mouse_left_held: bool = False

    # Scroll input (for zoom-around-pointer)
    scroll_delta: float = 0.0

    # One-shot command flags (reset after get_state)
    request_reload: bool = False
    request_reset: bool = False
    request_full_reset: bool = False
    toggle_recording: bool = False
    request_screenshot: bool = False
    request_world_size_change: bool = False

    # Config save/load (Ctrl+C/Ctrl+V)
    request_save_config: bool = False
    request_load_config: bool = False
    clipboard_text: str = ""  # For passing clipboard content to orchestrator

    # File save/load/delete (menu bar)
    request_save_file: bool = False
    request_load_file: bool = False
    request_delete_file: bool = False
    save_filename: str = ""  # Filename to save to (without extension)
    load_filename: str = ""  # Filename to load from (without extension)
    delete_filename: str = ""  # Filename to delete (without extension)
    load_watercolor_override: bool | None = None  # Override watercolor mode when loading

    # Config preview (for Load submenu hover)
    request_preview_config: bool = False  # Push rule for preview
    request_clear_preview: bool = False  # Pop preview rule
    preview_filename: str = ""  # Filename to preview

    # Rule history window flags
    request_preview_history_rule: bool = False
    request_clear_history_preview: bool = False
    request_load_history_rule: bool = False
    request_delete_history_rule: bool = False
    history_preview_index: int = -1
