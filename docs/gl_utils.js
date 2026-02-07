/**
 * WebGL 2 utility functions for shader compilation, texture/FBO creation, and uniforms.
 * Mirrors the role of gl_utils.py from the desktop version.
 */

export function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compilation failed:\n${info}`);
    }
    return shader;
}

export function createProgram(gl, vertSource, fragSource) {
    const vert = compileShader(gl, gl.VERTEX_SHADER, vertSource);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSource);
    const program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        gl.deleteShader(vert);
        gl.deleteShader(frag);
        throw new Error(`Program link failed:\n${info}`);
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return program;
}

export function createFloatTexture(gl, width, height, data = null) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
}

export function createFramebuffer(gl, texture) {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
}

// Cache for uniform locations to avoid repeated lookups
const uniformCache = new WeakMap();

function getUniformLocation(gl, program, name) {
    if (!uniformCache.has(program)) {
        uniformCache.set(program, {});
    }
    const cache = uniformCache.get(program);
    if (!(name in cache)) {
        cache[name] = gl.getUniformLocation(program, name);
    }
    return cache[name];
}

export function tryset(gl, program, name, value, typeHint) {
    const loc = getUniformLocation(gl, program, name);
    if (loc === null) return;

    if (typeof value === 'number') {
        if (typeHint === 'float' || !Number.isInteger(value)) {
            gl.uniform1f(loc, value);
        } else {
            gl.uniform1i(loc, value);
        }
    } else if (Array.isArray(value)) {
        if (value.length === 2) gl.uniform2f(loc, value[0], value[1]);
        else if (value.length === 3) gl.uniform3f(loc, value[0], value[1], value[2]);
        else if (value.length === 4) gl.uniform4f(loc, value[0], value[1], value[2], value[3]);
    }
}

export function setConfigUniforms(gl, program, config) {
    tryset(gl, program, 'config.cohorts', config.cohorts);
    tryset(gl, program, 'config.rule_seed', config.rule_seed);
    tryset(gl, program, 'config.sensor_gain', config.sensor_gain);
    tryset(gl, program, 'config.sensor_angle', config.sensor_angle);
    tryset(gl, program, 'config.sensor_distance', config.sensor_distance);
    tryset(gl, program, 'config.mutation_scale', config.mutation_scale);
    tryset(gl, program, 'config.global_force_mult', config.global_force_mult);
    tryset(gl, program, 'config.drag', config.drag);
    tryset(gl, program, 'config.strafe_power', config.strafe_power);
    tryset(gl, program, 'config.axial_force', config.axial_force);
    tryset(gl, program, 'config.lateral_force', config.lateral_force);
    tryset(gl, program, 'config.hazard_rate', config.hazard_rate);
    tryset(gl, program, 'config.trail_persistence', config.trail_persistence);
    tryset(gl, program, 'config.trail_diffusion', config.trail_diffusion);
}

export function setRuleUniforms(gl, program, rule) {
    for (let i = 0; i < 10; i++) {
        const base = i * 8;
        tryset(gl, program, `config_rule.centers[${i}].frequency`,
            [rule[base], rule[base + 1], rule[base + 2], rule[base + 3]]);
        tryset(gl, program, `config_rule.centers[${i}].amplitude`,
            [rule[base + 4], rule[base + 5], rule[base + 6], rule[base + 7]]);
    }
}

export function loadConfig(data) {
    const physics = data.physics;
    const settings = data.settings;
    return {
        cohorts: settings.num_cohorts,
        rule_seed: settings.rule_seed,
        sensor_gain: physics.sensor_gain,
        sensor_angle: physics.sensor_angle,
        sensor_distance: physics.sensor_distance,
        mutation_scale: physics.mutation_scale,
        global_force_mult: physics.global_force_mult,
        drag: physics.drag,
        strafe_power: physics.strafe_power,
        axial_force: physics.axial_force,
        lateral_force: physics.lateral_force,
        hazard_rate: physics.hazard_rate,
        trail_persistence: physics.trail_persistence,
        trail_diffusion: physics.trail_diffusion,
        rule: data.rule,
    };
}

export async function fetchShader(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load shader: ${path}`);
    return response.text();
}
