/**
 * Main orchestrator for Fluoddity WebGL 2 port.
 * Creates the WebGL context, particle system, and wires together
 * the state, input, UI, and logging modules.
 */

import { ParticleSystem } from './particle_system.js';
import { computeEntityRule } from './rule_utils.js';
import { RuleHistory, fetchConfig, createAppState } from './state.js';
import { setupKeyboard, setupMouse, screenToWorld } from './input.js';
import { setupUI, updateModeDisplay } from './ui.js';
import { createLogger } from './log.js';

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

    function selectParticleAt(clientX, clientY) {
        const world = screenToWorld(canvas, clientX, clientY);
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
        getCurrentPresetName: () => currentPresetName,
        setCurrentPresetName: (name) => { currentPresetName = name; },
        snapshotConfig: () => ({ ...config, rule: Array.from(config.rule) }),
        finalizePresetLoad(newConfig) {
            config = newConfig;
            system.setConfig(config);
            ruleHistory.push(config.rule, config.rule_seed);
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
    });

    setupMouse(canvas, state, {
        selectParticleAt,
        performUndo,
        setTrailDrawState: (s) => system.setTrailDrawState(s),
    });

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

    function frame() {
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

        system.renderDisplay();
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
