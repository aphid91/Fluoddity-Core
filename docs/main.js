/**
 * Main orchestrator for Fluoddity WebGL 2 port.
 * Creates the WebGL context, particle system, and wires together
 * the state, input, UI, and logging modules.
 */

import { ParticleSystem } from './particle_system.js';
import { computeEntityRule } from './rule_utils.js';
import { RuleHistory, fetchConfig, createAppState } from './state.js';
import { setupKeyboard, setupMouse, setupScroll, screenToWorld, updateCamera } from './input.js';
import { setupUI, updateModeDisplay, updateInitialConditionsDisplay, updateMutationScaleDisplay, updateCohortsDisplay } from './ui.js';
import { createLogger } from './log.js';
import { calibrate } from './calibrate.js';

// ─── Save/Load utilities (SIM7 cross-compatible format) ─────────────────────

async function zlibCompress(data) {
    const cs = new CompressionStream('deflate');
    const writer = cs.writable.getWriter();
    writer.write(data);
    writer.close();
    const chunks = [];
    const reader = cs.readable.getReader();
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}

async function zlibDecompress(data) {
    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    writer.write(data);
    writer.close();
    const chunks = [];
    const reader = ds.readable.getReader();
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}

function toUrlSafeBase64(bytes) {
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_');
}

function fromUrlSafeBase64(str) {
    let b64 = str.replace(/-/g, '+').replace(/_/g, '/');
    while (b64.length % 4 !== 0) {
        b64 += '=';
    }
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

// Default sweep/jitter keys matching Python's PHYSICS_PARAM_NAMES
const PARAM_NAMES = [
    'SENSOR_GAIN', 'SENSOR_ANGLE', 'SENSOR_DISTANCE', 'MUTATION_SCALE',
    'GLOBAL_FORCE_MULT', 'DRAG', 'AXIAL_FORCE', 'LATERAL_FORCE',
    'STRAFE_POWER', 'TRAIL_PERSISTENCE', 'TRAIL_DIFFUSION', 'HAZARD_RATE',
];

function defaultSweeps() {
    const d = {};
    for (const p of PARAM_NAMES) d[p] = 0.0;
    return d;
}

function defaultJitters() {
    const d = {};
    for (const p of PARAM_NAMES) d[p] = 0.0;
    return d;
}

function defaultSliderRanges() {
    return {
        "Sensor Gain": [0.0, 5.0, 0.0, 5.0],
        "Sensor Angle": [-1.0, 1.0, -1.0, 1.0],
        "Sensor Distance": [0.0, 4.0, 0.0, 4.0],
        "Mutation Scale": [-0.5, 0.5, -0.5, 0.5],
        "Global Force Mult": [0.0, 2.0, 0.0, 2.0],
        "Drag": [-1.0, 1.0, -1.0, 1.0],
        "Axial Force": [-1.0, 1.0, -1.0, 1.0],
        "Lateral Force": [-1.0, 1.0, -1.0, 1.0],
        "Strafe Power": [0.0, 0.5, 0.0, 0.5],
        "Trail Persistence": [0.0, 1.0, 0.0, 1.0],
        "Trail Diffusion": [0.0, 1.0, 0.0, 1.0],
        "Hazard Rate": [0.0, 0.05, 0.0, 0.05],
    };
}

async function main() {
    const canvas = document.getElementById('canvas');
    const logger = createLogger(document.getElementById('error-display'));

    // Size canvas to window
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();

    // Get WebGL 2 context
    const gl = canvas.getContext('webgl2', { antialias: false });
    if (!gl) {
        logger.error('WebGL 2 is not supported by your browser.');
        return;
    }
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
        logger.error('EXT_color_buffer_float extension not available. Float framebuffers are required.');
        return;
    }
    gl.getExtension('OES_texture_float_linear');

    // Load preset manifest
    let presetNames;
    try {
        const manifestResponse = await fetch('physics_configs/index.json', { cache: 'no-store' });
        if (!manifestResponse.ok) throw new Error(`HTTP ${manifestResponse.status}`);
        presetNames = await manifestResponse.json();
    } catch (e) {
        logger.error(`Failed to load preset manifest: ${e.message}`);
        return;
    }

    // ─── State ────────────────────────────────────────────────────────────────

    let currentPresetName = presetNames[0];
    let config;
    try {
        config = await fetchConfig(currentPresetName);
    } catch (e) {
        logger.error(`Failed to load config: ${e.message}`);
        return;
    }

    const state = createAppState();
    const ruleHistory = new RuleHistory();
    ruleHistory.push(config.rule, config.rule_seed);

    // ─── Particle system ──────────────────────────────────────────────────────

    let worldSize = parseFloat(document.getElementById('world-size-selector').value);
    const aspectRatio = window.innerWidth / window.innerHeight;
    const system = new ParticleSystem(gl, config, worldSize, aspectRatio);
    try {
        await system.init();
    } catch (e) {
        logger.error(`Initialization failed: ${e.message}`);
        console.error(e);
        return;
    }

    // ─── Actions (callbacks for input/UI modules) ─────────────────────────────

    function applyConfig(newConfig) {
        config = newConfig;
        system.setConfig(config);
        updateInitialConditionsDisplay(ui.el, config);
        updateMutationScaleDisplay(ui.el, config);
        updateCohortsDisplay(ui.el, config);
    }

    function applyRule(rule, seed) {
        config = { ...config, rule: Array.from(rule), rule_seed: seed };
        system.setConfig(config);
    }

    function performUndo() {
        const prev = ruleHistory.pop();
        if (prev) {
            applyRule(prev.rule, prev.seed);
        }
    }

    function toggleMode() {
        state.mouseMode = state.mouseMode === 'select' ? 'draw' : 'select';
        updateModeDisplay(ui.el, state);
        if (state.mouseMode === 'select') {
            system.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
        }
    }

    function togglePause() {
        state.paused = !state.paused;
    }

    function resetSimulation() {
        system.reset();
    }

    function randomizeSeed() {
        const current = ruleHistory.current();
        if (!current) return;
        const newSeed = Math.random();
        ruleHistory.push(current.rule, newSeed);
        applyRule(current.rule, newSeed);
    }

    function cycleInitialConditions() {
        const current = config.initial_conditions || 0;
        config.initial_conditions = (current + 1) % 3;
        system.setConfig(config);
        updateInitialConditionsDisplay(ui.el, config);
    }

    function selectParticleAt(clientX, clientY) {
        const cam = state.fancyCamera ? state.camera : null;
        const world = screenToWorld(canvas, clientX, clientY, cam, cam ? system.c : null);
        const data = system.readEntityData();
        const c = system.c;

        let bestDist = Infinity;
        let bestIndex = -1;
        for (let i = 0; i < c.entityCount; i++) {
            const base = i * 4;
            const dx = data[base] - world.x;
            const dy = data[base + 1] - world.y;
            const dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                bestIndex = i;
            }
        }
        if (bestIndex < 0) return;

        const entityRule = computeEntityRule(
            config.rule, config.rule_seed, config.mutation_scale,
            config.cohorts, bestIndex, c.entityCount
        );
        ruleHistory.push(Array.from(entityRule), config.rule_seed);
        applyRule(entityRule, config.rule_seed);
    }

    function changeWorldSize(newSize) {
        worldSize = newSize;
        const ar = window.innerWidth / window.innerHeight;
        system.reinitGPU(worldSize, ar);
        // Reset camera when world size changes
        state.camera.posX = 0;
        state.camera.posY = 0;
        state.camera.zoom = 1.0;
    }

    async function saveConfig() {
        const v7 = {
            version: 7,
            physics: {
                sensor_gain: config.sensor_gain,
                sensor_angle: config.sensor_angle,
                sensor_distance: config.sensor_distance,
                mutation_scale: config.mutation_scale,
                global_force_mult: config.global_force_mult,
                drag: config.drag,
                strafe_power: config.strafe_power,
                axial_force: config.axial_force,
                lateral_force: config.lateral_force,
                hazard_rate: config.hazard_rate,
                trail_persistence: config.trail_persistence,
                trail_diffusion: config.trail_diffusion,
            },
            slider_ranges: defaultSliderRanges(),
            sweeps: { x: defaultSweeps(), y: defaultSweeps(), cohort: defaultSweeps() },
            jitters: defaultJitters(),
            parameter_sweeps_enabled: false,
            settings: {
                disable_symmetry: false,
                absolute_orientation: 0,
                orientation_mix: 1.0,
                boundary_conditions: 0,
                initial_conditions: config.initial_conditions || 0,
                num_cohorts: config.cohorts,
                rule_seed: config.rule_seed,
            },
            appearance: {
                ink_weight: 1.0,
                hue_sensitivity: 0.5,
                color_by_cohort: true,
                watercolor_mode: false,
                emboss_mode: 0,
                emboss_intensity: 0.5,
                emboss_smoothness: 0.1,
            },
            rule: Array.from(config.rule),
            notes: "",
        };
        try {
            const jsonBytes = new TextEncoder().encode(JSON.stringify(v7));
            const compressed = await zlibCompress(jsonBytes);
            const saveString = 'SIM7:' + toUrlSafeBase64(compressed);
            await navigator.clipboard.writeText(saveString);
            console.log(`Config saved to clipboard (${saveString.length} chars)`);
        } catch (err) {
            logger.error(`Save failed: ${err.message}`);
        }
    }

    async function loadConfigFromClipboard() {
        try {
            const text = (await navigator.clipboard.readText()).trim();
            if (!text.startsWith('SIM')) {
                logger.error('Clipboard does not contain a valid Fluoddity config');
                return;
            }
            const colonIdx = text.indexOf(':');
            if (colonIdx === -1) {
                logger.error('Invalid config format');
                return;
            }
            const version = parseInt(text.substring(3, colonIdx), 10);
            if (version !== 7) {
                logger.error(`Unsupported config version: ${version} (expected 7)`);
                return;
            }
            const encoded = text.substring(colonIdx + 1);
            const compressed = fromUrlSafeBase64(encoded);
            const jsonBytes = await zlibDecompress(compressed);
            const data = JSON.parse(new TextDecoder().decode(jsonBytes));

            const newConfig = {
                cohorts: data.settings.num_cohorts,
                rule_seed: data.settings.rule_seed,
                sensor_gain: data.physics.sensor_gain,
                sensor_angle: data.physics.sensor_angle,
                sensor_distance: data.physics.sensor_distance,
                mutation_scale: data.physics.mutation_scale,
                global_force_mult: data.physics.global_force_mult,
                drag: data.physics.drag,
                strafe_power: data.physics.strafe_power,
                axial_force: data.physics.axial_force,
                lateral_force: data.physics.lateral_force,
                hazard_rate: data.physics.hazard_rate,
                trail_persistence: data.physics.trail_persistence,
                trail_diffusion: data.physics.trail_diffusion,
                rule: data.rule,
                initial_conditions: data.settings.initial_conditions !== undefined ? data.settings.initial_conditions : 0,
            };

            applyConfig(newConfig);
            ruleHistory.reset(newConfig.rule, newConfig.rule_seed);
            ui.el.presetTrigger.textContent = '(Pasted)';
            currentPresetName = null;
            console.log('Config loaded from clipboard');
        } catch (err) {
            logger.error(`Load failed: ${err.message}`);
        }
    }

    // Dropdown needs a way to register its closeDropdown for keyboard use
    let closeDropdownFn = null;

    // ─── Wire up UI ───────────────────────────────────────────────────────────

    const ui = setupUI(presetNames, currentPresetName, state, {
        toggleMode,
        performUndo,
        resetSimulation,
        applyConfig,
        changeWorldSize,
        cycleInitialConditions,
        onMutationScaleChange(value) {
            config.mutation_scale = value;
            system.setConfig(config);
        },
        onCohortsChange(value) {
            config.cohorts = value;
            system.setConfig(config);
        },
        getCurrentPresetName: () => currentPresetName,
        setCurrentPresetName: (name) => { currentPresetName = name; },
        snapshotConfig: () => ({ ...config, rule: Array.from(config.rule) }),
        finalizePresetLoad(newConfig) {
            config = newConfig;
            system.setConfig(config);
            ruleHistory.reset(config.rule, config.rule_seed);
            updateInitialConditionsDisplay(ui.el, config);
            updateMutationScaleDisplay(ui.el, config);
            updateCohortsDisplay(ui.el, config);
        },
        registerCloseDropdown(fn) { closeDropdownFn = fn; },
        logError: (msg) => logger.error(msg),
    });

    // ─── Wire up input ───────────────────────────────────────────────────────

    setupKeyboard(state, {
        toggleMode,
        togglePause,
        resetSimulation,
        randomizeSeed,
        closeDropdown: (restore) => closeDropdownFn && closeDropdownFn(restore),
        saveConfig,
        loadConfig: loadConfigFromClipboard,
    });

    setupMouse(canvas, state, {
        selectParticleAt,
        performUndo,
        setTrailDrawState: (s) => system.setTrailDrawState(s),
        getConstants: () => system.c,
    });

    setupScroll(canvas, state);

    // ─── Auto-detect optimal settings ────────────────────────────────────────

    {
        const ar = window.innerWidth / window.innerHeight;
        const optimal = calibrate(gl, system, ar);
        worldSize = optimal.worldSize;

        // Update UI to reflect calibrated values
        // Note: String(1.0) produces "1" which doesn't match option value "1.0",
        // so we match by numeric value instead.
        const wsOpts = ui.el.worldSizeSelector.options;
        for (let i = 0; i < wsOpts.length; i++) {
            if (parseFloat(wsOpts[i].value) === optimal.worldSize) {
                ui.el.worldSizeSelector.selectedIndex = i;
                break;
            }
        }
        ui.el.physicsFreqSlider.value = String(optimal.physicsSpeed);
        ui.el.physicsFreqValue.textContent = String(optimal.physicsSpeed);
    }
    document.getElementById('calibration-overlay').style.display = 'none';

    // Sync new controls with initially loaded config
    updateInitialConditionsDisplay(ui.el, config);
    updateMutationScaleDisplay(ui.el, config);
    updateCohortsDisplay(ui.el, config);

    // ─── Resize ───────────────────────────────────────────────────────────────

    let resizeTimeout = null;
    window.addEventListener('resize', () => {
        resize();
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const ar = window.innerWidth / window.innerHeight;
            system.reinitGPU(worldSize, ar);
        }, 300);
    });

    // ─── Render loop ──────────────────────────────────────────────────────────

    let lastFrameTime = performance.now();

    function frame() {
        const now = performance.now();
        const dt = Math.min((now - lastFrameTime) / 1000.0, 0.05);
        lastFrameTime = now;

        updateCamera(state, dt);

        if (!state.paused) {
            const physicsFrequency = parseInt(ui.el.physicsFreqSlider.value, 10);

            if (state.mouseMode === 'draw' && state.mouseDown) {
                system.setTrailDrawState({
                    x: state.mousePos.x,
                    y: state.mousePos.y,
                    prevX: state.prevMousePos.x,
                    prevY: state.prevMousePos.y,
                    radius: parseFloat(ui.el.drawSizeSlider.value),
                    power: parseFloat(ui.el.drawPowerSlider.value),
                });
            }

            for (let i = 0; i < physicsFrequency; i++) {
                system.advance();
            }

            if (state.mouseMode === 'draw' && state.mouseDown) {
                state.prevMousePos = { ...state.mousePos };
            }

            if (state.mouseMode === 'draw' && !state.mouseDown) {
                system.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
            }
        }

        const brightness = parseFloat(ui.el.brightnessSlider.value);
        system.renderDisplay(state.fancyCamera, state.camera, brightness);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
