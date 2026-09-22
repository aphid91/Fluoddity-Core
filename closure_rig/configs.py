"""Config discovery, de-duplication, and the Core-semantics compatibility audit.

Configs live in three places now:

  physics_configs/            -- this repo (already includes Fluoddity's Core set)
  docs/physics_configs/       -- the WebGL port's presets
  <Fluoddity>/physics_configs/Core/  -- the sibling repo, if checked out

The sibling repo is optional for configs (its Core set was vendored into
physics_configs/), but its shaders are what document the settings Core ignores.
"""

import glob
import hashlib
import json
import os

# What Core's shaders actually do, regardless of what a config asks for.
# entity_update.glsl always wraps position, always orients relative to velocity,
# and always mirror-symmetrises the rule (see calculate_entity_behavior).
CORE_SEMANTICS = {
    'boundary_conditions': 2,     # 2 == wrap
    'absolute_orientation': 0,    # 0 == relative to velocity
    'disable_symmetry': False,    # Core always symmetrises
}
# initial_conditions and orientation_mix are handled separately: Core now honours
# initial_conditions, and orientation_mix is inert when absolute_orientation == 0.

# Configs held out of experiment sweeps. Their E0 results are kept; they are
# simply not carried into E1 onward. Pass --include-excluded to override.
EXCLUDED = {
    'Pop': 'E0.4 warm-up floor-limited: the drift test passed on its first and '
           'only evaluation (drift 0.071 vs noise floor 0.267), so its settling '
           'time is unmeasured and later experiments cannot be started from a '
           'known-settled state.',
}

SOURCES = [
    ('core', 'physics_configs/*.json'),
    ('docs', 'docs/physics_configs/*.json'),
]
FLUODDITY_GLOB = 'physics_configs/Core/*.json'


def _physics_signature(data):
    """Hash of the physics values, so the same preset in two folders dedupes."""
    phys = {k: round(float(v), 6) for k, v in sorted(data.get('physics', {}).items())}
    return hashlib.md5(json.dumps(phys).encode()).hexdigest()[:12]


def discover(repo_root='.', fluoddity_path=None):
    """Every config across all sources, de-duplicated by physics signature.

    Returns a list of dicts, each with path/name/source/signature/settings.
    First source to supply a name or signature wins, so local copies are preferred.
    """
    found, seen_sig, seen_name = [], {}, {}
    patterns = [(src, os.path.join(repo_root, pat)) for src, pat in SOURCES]
    if fluoddity_path and os.path.isdir(fluoddity_path):
        patterns.append(('fluoddity', os.path.join(fluoddity_path, FLUODDITY_GLOB)))

    for source, pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            name = os.path.splitext(os.path.basename(path))[0]
            if name == 'index':          # docs/ ships a manifest, not a config
                continue
            try:
                data = json.load(open(path))
            except Exception as e:
                print(f"  skipping unreadable config {path}: {e}")
                continue
            if 'physics' not in data or 'settings' not in data:
                continue
            sig = _physics_signature(data)
            # Dedupe on BOTH name and physics: two configs sharing a name would
            # otherwise collide in results/<name>/, and the second would silently
            # overwrite the first.
            dup = seen_sig.get(sig) or seen_name.get(name)
            if dup is not None:
                dup['also_in'].append(f"{source}/{name}")
                continue
            entry = {
                'name': name,
                'path': path,
                'source': source,
                'signature': sig,
                'excluded': name in EXCLUDED,
                'excluded_reason': EXCLUDED.get(name, ''),
                'also_in': [],
                'settings': data['settings'],
                'physics': data['physics'],
            }
            seen_sig[sig] = entry
            seen_name[name] = entry
            found.append(entry)
    return found


def audit_row(entry):
    """One row of results/config_audit.csv.

    A config is flagged core_semantics_approximation when it asks for behaviour
    Core does not implement, meaning its results describe a related-but-different
    system. absolute_orientation is called out separately because a non-relative
    heading would break the rotational isotropy that E3's augmentation relies on.
    """
    s = entry['settings']
    p = entry['physics']
    diffs = [k for k, core_val in CORE_SEMANTICS.items() if s.get(k, core_val) != core_val]
    persistence = min(float(p.get('trail_persistence', 0.0)), 0.999)
    return {
        'name': entry['name'],
        'source': entry['source'],
        'excluded': entry.get('excluded', False),
        'excluded_reason': entry.get('excluded_reason', ''),
        'signature': entry['signature'],
        'also_in': ';'.join(entry['also_in']),
        'boundary_conditions': s.get('boundary_conditions', ''),
        'initial_conditions': s.get('initial_conditions', ''),
        'absolute_orientation': s.get('absolute_orientation', ''),
        'orientation_mix': s.get('orientation_mix', ''),
        'disable_symmetry': s.get('disable_symmetry', ''),
        'differs_from_core': ';'.join(diffs),
        'core_semantics_approximation': bool(diffs),
        'breaks_e3_isotropy': s.get('absolute_orientation', 0) != 0,
        'num_cohorts': s.get('num_cohorts', ''),
        'trail_persistence': p.get('trail_persistence', ''),
        'memory_time_steps': round(1.0 / (1.0 - persistence), 2),
        'trail_diffusion': p.get('trail_diffusion', ''),
        'drag': p.get('drag', ''),
        'negative_drag': float(p.get('drag', 0.0)) < 0,
        'hazard_rate': p.get('hazard_rate', ''),
        'sensor_distance': p.get('sensor_distance', ''),
    }
