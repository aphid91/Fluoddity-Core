#!/usr/bin/env python3
"""
Script to update slider_ranges in physics config JSON files.

Updates the first pair of slider range values to accommodate the current physics value
if it's outside the base range, while keeping the second pair exactly as the base range.
"""

import json
import os
from pathlib import Path

# Base ranges for each parameter
BASE_RANGES = {
    "Axial Force": [-1.0, 1.0],
    "Lateral Force": [-1.0, 1.0],
    "Sensor Gain": [0.0, 10.0],
    "Mutation Scale": [0.0, 1.0],
    "Drag": [-1.0, 1.0],
    "Strafe Power": [0.0, 0.5],
    "Sensor Angle": [-1.0, 1.0],
    "Global Force Mult": [0.0, 2.0],
    "Sensor Distance": [0.0, 3.0],
    "Trail Persistence": [0.0, 1.0],
    "Trail Diffusion": [0.0, 1.0],
    "Hazard Rate": [0.0, 0.05],
}

# Mapping from slider label to physics field name
LABEL_TO_FIELD = {
    "Axial Force": "axial_force",
    "Lateral Force": "lateral_force",
    "Sensor Gain": "sensor_gain",
    "Mutation Scale": "mutation_scale",
    "Drag": "drag",
    "Strafe Power": "strafe_power",
    "Sensor Angle": "sensor_angle",
    "Global Force Mult": "global_force_mult",
    "Sensor Distance": "sensor_distance",
    "Trail Persistence": "trail_persistence",
    "Trail Diffusion": "trail_diffusion",
    "Hazard Rate": "hazard_rate",
}


def update_slider_range(label, physics_value, base_range):
    """
    Calculate the new slider range for a parameter.

    Args:
        label: The parameter label (e.g., "Sensor Gain")
        physics_value: The current value from the physics object
        base_range: The base range [min, max]

    Returns:
        A list of 4 values: [extended_min, extended_max, base_min, base_max]
    """
    base_min, base_max = base_range

    # Start with the base range for the first pair
    extended_min = base_min
    extended_max = base_max

    # Extend if the physics value is outside the base range
    if physics_value < base_min:
        extended_min = physics_value
    elif physics_value > base_max:
        extended_max = physics_value

    # Return [extended_min, extended_max, base_min, base_max]
    return [extended_min, extended_max, base_min, base_max]


def update_json_file(file_path):
    """
    Update the slider_ranges in a single JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        True if file was modified, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if this file has the expected structure
    if 'physics' not in data or 'slider_ranges' not in data:
        print(f"  Skipping {file_path.name} - missing physics or slider_ranges")
        return False

    modified = False

    # Update each slider range
    for label, base_range in BASE_RANGES.items():
        if label not in data['slider_ranges']:
            print(f"  Warning: {file_path.name} missing slider range for '{label}'")
            continue

        # Get the physics field name
        field_name = LABEL_TO_FIELD[label]
        if field_name not in data['physics']:
            print(f"  Warning: {file_path.name} missing physics field '{field_name}'")
            continue

        physics_value = data['physics'][field_name]
        old_range = data['slider_ranges'][label]
        new_range = update_slider_range(label, physics_value, base_range)

        # Check if it changed
        if old_range != new_range:
            data['slider_ranges'][label] = new_range
            modified = True
            print(f"  {label}: {old_range} -> {new_range}")

    # Write back if modified
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True

    return False


def main():
    """Main function to process all JSON files."""
    script_dir = Path(__file__).parent

    # Directories to process
    directories = [
        script_dir / "physics_configs" / "Core",
        script_dir / "physics_configs" / "Advanced",
    ]

    total_files = 0
    modified_files = 0

    for directory in directories:
        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            continue

        print(f"\nProcessing directory: {directory.name}")
        print("-" * 60)

        # Get all JSON files
        json_files = sorted(directory.glob("*.json"))

        for json_file in json_files:
            total_files += 1
            print(f"\n{json_file.name}:")

            if update_json_file(json_file):
                modified_files += 1
            else:
                print("  No changes needed")

    print("\n" + "=" * 60)
    print(f"Summary: Modified {modified_files} out of {total_files} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
