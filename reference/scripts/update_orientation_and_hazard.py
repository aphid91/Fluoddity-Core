"""
Migration script to update existing physics configs for new orientation and hazard rate features.

Changes:
1. Convert absolute_orientation from boolean to int (false -> 0, true -> 1)
2. Add orientation_mix field (default 1.0)
3. Move hazard_rate from simulation settings to physics params
4. Add "Hazard Rate" to slider_ranges
5. Add "HAZARD_RATE" to x_sweeps, y_sweeps, cohort_sweeps

Usage:
    python scripts/update_orientation_and_hazard.py [--dry-run]
"""

import json
from pathlib import Path
import argparse


def migrate_config_file(config_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Migrate a single config file.

    Args:
        config_path: Path to the config file
        dry_run: If True, don't write changes

    Returns:
        (modified, message) tuple
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to read: {e}"

    modified = False
    changes = []

    # 1. Convert absolute_orientation boolean -> int
    settings = data.get('settings', {})
    if 'absolute_orientation' in settings:
        abs_orient = settings['absolute_orientation']
        if isinstance(abs_orient, bool):
            settings['absolute_orientation'] = 1 if abs_orient else 0
            modified = True
            changes.append(f"absolute_orientation: {abs_orient} -> {settings['absolute_orientation']}")

    # 2. Add orientation_mix if missing
    if 'orientation_mix' not in settings:
        settings['orientation_mix'] = 1.0
        modified = True
        changes.append("Added orientation_mix: 1.0")

    # 3. Move hazard_rate to physics params if it's in settings
    # (For configs created after our changes, it will already be in physics)
    physics = data.get('physics', {})
    if 'hazard_rate' not in physics:
        hazard_rate = settings.get('hazard_rate', 0.0)
        physics['hazard_rate'] = hazard_rate
        modified = True
        changes.append(f"Moved hazard_rate to physics: {hazard_rate}")

    # Remove hazard_rate from settings if present (it's now in physics)
    if 'hazard_rate' in settings:
        del settings['hazard_rate']

    # 4. Add "Hazard Rate" to slider_ranges if missing
    slider_ranges = data.get('slider_ranges', {})
    if 'Hazard Rate' not in slider_ranges:
        slider_ranges['Hazard Rate'] = [0.0, 0.05, 0.0, 0.05]
        modified = True
        changes.append("Added 'Hazard Rate' to slider_ranges")

    # 5. Add "HAZARD_RATE" to sweep dicts if missing
    sweeps = data.get('sweeps', {})
    for sweep_key in ['x', 'y', 'cohort']:
        sweep_dict = sweeps.get(sweep_key, {})
        if 'HAZARD_RATE' not in sweep_dict:
            sweep_dict[sweep_key] = 0.0
            sweeps[sweep_key] = sweep_dict
            modified = True
            changes.append(f"Added 'HAZARD_RATE' to {sweep_key}_sweeps")

    # Update data dict
    data['settings'] = settings
    data['physics'] = physics
    data['slider_ranges'] = slider_ranges
    data['sweeps'] = sweeps

    # Write back if modified and not dry run
    if modified and not dry_run:
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            return False, f"Failed to write: {e}"

    if modified:
        return True, "; ".join(changes)
    else:
        return False, "Already up to date"


def main():
    parser = argparse.ArgumentParser(
        description="Migrate physics configs for new orientation and hazard rate features"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be changed without modifying files"
    )
    args = parser.parse_args()

    # Find all JSON config files
    config_dir = Path('physics_configs')
    if not config_dir.exists():
        print(f"Error: Config directory '{config_dir}' not found")
        print("Make sure you're running this script from the SimScratch root directory")
        return

    config_files = list(config_dir.glob('*.json'))
    if not config_files:
        print(f"No JSON config files found in {config_dir}")
        return

    print(f"Found {len(config_files)} config file(s)")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    else:
        print()

    modified_count = 0
    error_count = 0

    for config_file in sorted(config_files):
        modified, message = migrate_config_file(config_file, dry_run=args.dry_run)

        if "Failed" in message:
            print(f"❌ {config_file.name}: {message}")
            error_count += 1
        elif modified:
            print(f"✓ {config_file.name}: {message}")
            modified_count += 1
        else:
            print(f"  {config_file.name}: {message}")

    print()
    print(f"Summary:")
    print(f"  Total configs: {len(config_files)}")
    print(f"  Modified: {modified_count}")
    print(f"  Already current: {len(config_files) - modified_count - error_count}")
    print(f"  Errors: {error_count}")

    if args.dry_run and modified_count > 0:
        print()
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
