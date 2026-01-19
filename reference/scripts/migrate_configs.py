#!/usr/bin/env python3
"""
Migration script to convert legacy SIM1-6 config files to new JSON format (SIM7).

Usage:
    python scripts/migrate_configs.py [--delete-old]

Options:
    --delete-old    Delete the old .txt files after successful conversion
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from services
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.config_saver import ConfigSaver


def migrate_configs(configs_dir: Path, delete_old: bool = False) -> tuple[int, int, list[str]]:
    """
    Migrate all .txt config files to .json format.

    Returns:
        (success_count, fail_count, errors)
    """
    saver = ConfigSaver()
    success = 0
    failed = 0
    errors = []

    txt_files = list(configs_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} .txt files to migrate")

    for txt_path in txt_files:
        try:
            # Read the old config
            content = txt_path.read_text()

            # Check if it's already JSON (shouldn't be in .txt, but just in case)
            if content.strip().startswith('{'):
                print(f"  SKIP {txt_path.name} (already JSON)")
                continue

            # Decode using legacy format
            config = saver.decode_legacy(content)
            if config is None:
                errors.append(f"{txt_path.name}: Failed to decode legacy format")
                failed += 1
                continue

            # Save as JSON with same base name
            json_path = txt_path.with_suffix('.json')
            saver.save_to_file(config, json_path)
            print(f"  OK   {txt_path.name} -> {json_path.name}")
            success += 1

            # Optionally delete old file
            if delete_old:
                txt_path.unlink()
                print(f"       Deleted {txt_path.name}")

        except Exception as e:
            errors.append(f"{txt_path.name}: {e}")
            failed += 1

    return success, failed, errors


def main():
    delete_old = "--delete-old" in sys.argv

    # Find physics_configs directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    configs_dir = project_dir / "physics_configs"

    if not configs_dir.exists():
        print(f"Error: {configs_dir} does not exist")
        sys.exit(1)

    print(f"Migrating configs in {configs_dir}")
    print(f"Delete old files: {delete_old}")
    print()

    success, failed, errors = migrate_configs(configs_dir, delete_old)

    print()
    print(f"Results: {success} succeeded, {failed} failed")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
