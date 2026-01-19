from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class PreferencesState:
    """User preferences that persist between program sessions."""

    # Camera/rendering preferences
    speedmult: int = 1
    motion_blur: bool = True
    blur_quality: int = 1  # Motion blur render cadence (1 = every frame, 2 = every 2 frames, etc.)
    world_size: float = 1.0  # World size multiplier (affects entity count and canvas dimensions)
    rule_seed: float = 0.0
    brightness: float = 1.0  # Global brightness multiplier
    exposure: float = 0.0  # Frame blending for motion blur effect (0=disabled, 1=long exposure)

    # UI preferences
    show_preferences_window: bool = True  # Whether preferences window is visible
    show_controls_window: bool = False  # Help controls window
    show_parameter_sweeps_window: bool = False  # Help parameter sweeps window
    show_tutorial_window: bool = True  # Help tutorial window
    show_performance_window: bool = False  # Help performance window
    physics_tooltips_enabled: bool = True
    debug_arrows: bool = False  # Visual debug overlay for velocity field
    arrow_sensitivity: float = 9.0  # Velocity scale for debug arrows (pow(2, x))
    mouse_mode: str = "Select Particle"  # "Select Particle" or "Draw Trail"
    draw_size: float = 0.1  # Gaussian kernel width for trail drawing
    draw_power: float = 1.0  # Velocity strength when drawing trails
    menu_close_threshold: float = 80.0  # Distance in pixels before menus auto-close

    # Physics slider group collapsed states (True = expanded/open, False = collapsed)
    physics_group_basics: bool = True  # Default: open (trail sensors + mutation)
    physics_group_forces: bool = True  # Default: open (global force mult, drag)
    physics_group_advanced: bool = False  # Default: collapsed

    # Load menu collapsed states (True = expanded/open, False = collapsed)
    load_menu_core_open: bool = True  # Default: open
    load_menu_custom_open: bool = True  # Default: open
    load_menu_advanced_open: bool = False  # Default: collapsed

    # Recording preferences
    max_frames: int = 1800  # 150 * 12
    motion_blur_samples: int = 12
    supersample_k: int = 2
    filename_prefix: str = ""
    recording_motion_blur: bool = True  # Motion blur setting used during video recording
    recording_blur_quality: int = 1  # Blur quality setting used during video recording


def save_preferences(prefs: PreferencesState, filepath: Path | str = "preferences.config") -> None:
    """Save preferences to a JSON file."""
    filepath = Path(filepath)
    data = asdict(prefs)
    filepath.write_text(json.dumps(data, indent=2))


def load_preferences(filepath: Path | str = "preferences.config") -> PreferencesState:
    """Load preferences from a JSON file. Returns default preferences if file doesn't exist."""
    filepath = Path(filepath)
    if not filepath.exists():
        return PreferencesState()

    try:
        data = json.loads(filepath.read_text())
        # Filter out any fields that are no longer in PreferencesState (backward compat)
        valid_fields = set(PreferencesState.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return PreferencesState(**filtered_data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Failed to load preferences from {filepath}: {e}")
        print("Using default preferences")
        return PreferencesState()
