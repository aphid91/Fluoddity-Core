import json
import moderngl


def load_config(path: str) -> dict:
    """Load config from JSON file and return the physics values needed by shaders."""
    with open(path, 'r') as f:
        data = json.load(f)

    physics = data['physics']
    settings = data['settings']

    # Parse rule: 80 floats -> 10 FourierCenters, each with frequency(4) + amplitude(4)
    rule = data['rule']

    return {
        'cohorts': settings['num_cohorts'],
        'rule_seed': settings['rule_seed'],
        'sensor_gain': physics['sensor_gain'],
        'sensor_angle': physics['sensor_angle'],
        'sensor_distance': physics['sensor_distance'],
        'mutation_scale': physics['mutation_scale'],
        'global_force_mult': physics['global_force_mult'],
        'drag': physics['drag'],
        'strafe_power': physics['strafe_power'],
        'axial_force': physics['axial_force'],
        'lateral_force': physics['lateral_force'],
        'hazard_rate': physics['hazard_rate'],
        'trail_persistence': physics['trail_persistence'],
        'trail_diffusion': physics['trail_diffusion'],
        'rule': rule,
    }


def set_rule_uniform(program: moderngl.Program, rule: list):
    """Set the Rule uniform (10 FourierCenters, each with frequency vec4 + amplitude vec4)."""
    for i in range(10):
        base = i * 8
        tryset(program, f'config_rule.centers[{i}].frequency', tuple(rule[base:base+4]))
        tryset(program, f'config_rule.centers[{i}].amplitude', tuple(rule[base+4:base+8]))


def set_config_uniform(program: moderngl.Program, config: dict):
    """Set all ConfigData struct uniforms on a program."""
    tryset(program, 'config.cohorts', config['cohorts'])
    tryset(program, 'config.rule_seed', config['rule_seed'])
    tryset(program, 'config.sensor_gain', config['sensor_gain'])
    tryset(program, 'config.sensor_angle', config['sensor_angle'])
    tryset(program, 'config.sensor_distance', config['sensor_distance'])
    tryset(program, 'config.mutation_scale', config['mutation_scale'])
    tryset(program, 'config.global_force_mult', config['global_force_mult'])
    tryset(program, 'config.drag', config['drag'])
    tryset(program, 'config.strafe_power', config['strafe_power'])
    tryset(program, 'config.axial_force', config['axial_force'])
    tryset(program, 'config.lateral_force', config['lateral_force'])
    tryset(program, 'config.hazard_rate', config['hazard_rate'])
    tryset(program, 'config.trail_persistence', config['trail_persistence'])
    tryset(program, 'config.trail_diffusion', config['trail_diffusion'])


def read_shader(path: str):
    result = ""
    with open(path, 'r') as file:
        result = file.read()
    return result


MUTED_TRYSET_WARNINGS = {}


def tryset(program: moderngl.Program, uniform, value):
    """
    Gracefully handle a uniform that doesn't appear in program.
    Uniforms are frequently optimized out if they are not used in the current version of the shader.
    """
    if uniform in program:
        program[uniform] = value
    else:
        global MUTED_TRYSET_WARNINGS
        if uniform not in MUTED_TRYSET_WARNINGS:
            MUTED_TRYSET_WARNINGS[uniform] = 0
        MUTED_TRYSET_WARNINGS[uniform] += 1
        if MUTED_TRYSET_WARNINGS[uniform] < 10:
            print('Warning: ', uniform, ' not present in ', program)
