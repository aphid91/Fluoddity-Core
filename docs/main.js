/**
 * Main entry point for Fluoddity WebGL 2 port.
 * Sets up canvas, WebGL context, loads config, and runs the render loop.
 * Features: physics frequency control, particle selection, trail drawing,
 * undo history, and config preview-on-hover dropdown.
 */

import { loadConfig } from './gl_utils.js';
import { ParticleSystem } from './particle_system.js';
import { computeEntityRule } from './rule_utils.js';

// ─── Rule History (undo stack) ───────────────────────────────────────────────

class RuleHistory {
    constructor(maxSize = 200) {
        this.stack = []; // Array of {rule: number[], seed: number}
        this.maxSize = maxSize;
    }

    push(rule, seed) {
        this.stack.push({ rule: Array.from(rule), seed });
        if (this.stack.length > this.maxSize) {
            this.stack.shift();
        }
    }

    pop() {
        if (this.stack.length <= 1) return null;
        this.stack.pop(); // remove current
        return this.stack[this.stack.length - 1]; // return previous (now current)
    }

    current() {
        return this.stack.length > 0 ? this.stack[this.stack.length - 1] : null;
    }

    get length() {
        return this.stack.length;
    }
}

// ─── Config cache (for preview hover) ────────────────────────────────────────

const configCache = new Map();

async function fetchConfig(name) {
    if (configCache.has(name)) return configCache.get(name);
    const path = `physics_configs/${name}.json`;
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${path}`);
    const data = await response.json();
    const config = loadConfig(data);
    configCache.set(name, config);
    return config;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
    const canvas = document.getElementById('canvas');
    const errorDisplay = document.getElementById('error-display');

    // Size canvas to window
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();

    // Get WebGL 2 context
    const gl = canvas.getContext('webgl2', { antialias: false });
    if (!gl) {
        showError('WebGL 2 is not supported by your browser.');
        return;
    }

    // Required extension for rendering to float textures
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
        showError('EXT_color_buffer_float extension not available. Float framebuffers are required.');
        return;
    }

    // Load preset manifest
    let presetNames;
    try {
        const manifestResponse = await fetch('physics_configs/index.json', { cache: 'no-store' });
        if (!manifestResponse.ok) throw new Error(`HTTP ${manifestResponse.status}`);
        presetNames = await manifestResponse.json();
    } catch (e) {
        showError(`Failed to load preset manifest: ${e.message}`);
        return;
    }

    // ─── UI elements ─────────────────────────────────────────────────────────

    const worldSizeSelector = document.getElementById('world-size-selector');
    const resetButton = document.getElementById('reset-button');
    const undoButton = document.getElementById('undo-button');
    const physicsFreqSlider = document.getElementById('physics-freq');
    const physicsFreqValue = document.getElementById('physics-freq-value');
    const modeToggle = document.getElementById('mode-toggle');
    const drawControls = document.getElementById('draw-controls');
    const drawSizeSlider = document.getElementById('draw-size');
    const drawSizeValue = document.getElementById('draw-size-value');
    const drawPowerSlider = document.getElementById('draw-power');
    const drawPowerValue = document.getElementById('draw-power-value');
    const presetDropdown = document.getElementById('preset-dropdown');
    const presetTrigger = document.getElementById('preset-trigger');
    const presetMenu = document.getElementById('preset-menu');

    // ─── Custom dropdown setup ───────────────────────────────────────────────

    let currentPresetName = presetNames[0];

    // Populate dropdown items
    for (const name of presetNames) {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.dataset.preset = name;
        item.textContent = name.replace(/([A-Z])/g, ' $1').trim();
        presetMenu.appendChild(item);
    }

    function formatPresetName(name) {
        return name.replace(/([A-Z])/g, ' $1').trim();
    }

    // Mark the active item
    function updateActiveItem() {
        for (const item of presetMenu.children) {
            item.classList.toggle('active', item.dataset.preset === currentPresetName);
        }
    }

    // ─── Load default config ─────────────────────────────────────────────────

    let config;
    try {
        config = await fetchConfig(currentPresetName);
    } catch (e) {
        showError(`Failed to load config: ${e.message}`);
        return;
    }

    presetTrigger.textContent = formatPresetName(currentPresetName);
    updateActiveItem();

    // ─── Create particle system ──────────────────────────────────────────────

    let worldSize = parseFloat(worldSizeSelector.value);
    const aspectRatio = window.innerWidth / window.innerHeight;
    const system = new ParticleSystem(gl, config, worldSize, aspectRatio);
    try {
        await system.init();
    } catch (e) {
        showError(`Initialization failed: ${e.message}`);
        console.error(e);
        return;
    }

    // ─── State ───────────────────────────────────────────────────────────────

    const ruleHistory = new RuleHistory();
    ruleHistory.push(config.rule, config.rule_seed);

    let mouseMode = 'select'; // 'select' or 'draw'
    let mouseDown = false;
    let mousePos = { x: 0, y: 0 }; // canvas UV [0,1]
    let prevMousePos = { x: 0, y: 0 }; // previous frame's mouse UV

    // Preview state
    let previewActive = false;
    let previewBaseConfig = null; // full config object saved before preview
    let dropdownOpen = false;
    let previewGeneration = 0; // guards against async race conditions in hover preview

    // ─── Helper: apply a rule+seed to the running system ─────────────────────

    function applyConfig(newConfig) {
        config = newConfig;
        system.setConfig(config);
    }

    function applyRule(rule, seed) {
        config = { ...config, rule: Array.from(rule), rule_seed: seed };
        system.setConfig(config);
    }

    // ─── Undo ────────────────────────────────────────────────────────────────

    function performUndo() {
        const prev = ruleHistory.pop();
        if (prev) {
            applyRule(prev.rule, prev.seed);
        }
    }

    undoButton.addEventListener('click', performUndo);

    // ─── World size ──────────────────────────────────────────────────────────

    worldSizeSelector.addEventListener('change', () => {
        worldSize = parseFloat(worldSizeSelector.value);
        const ar = window.innerWidth / window.innerHeight;
        system.reinitGPU(worldSize, ar);
    });

    // ─── Reset ───────────────────────────────────────────────────────────────

    resetButton.addEventListener('click', () => {
        system.reset();
    });

    // ─── Physics frequency slider ────────────────────────────────────────────

    physicsFreqSlider.addEventListener('input', () => {
        physicsFreqValue.textContent = physicsFreqSlider.value;
    });

    // ─── Draw parameter sliders ──────────────────────────────────────────────

    drawSizeSlider.addEventListener('input', () => {
        drawSizeValue.textContent = parseFloat(drawSizeSlider.value).toFixed(3);
    });
    drawPowerSlider.addEventListener('input', () => {
        drawPowerValue.textContent = parseFloat(drawPowerSlider.value).toFixed(2);
    });

    // ─── Mode toggle ─────────────────────────────────────────────────────────

    function setMode(mode) {
        mouseMode = mode;
        if (mode === 'draw') {
            modeToggle.textContent = 'Draw Trail (T)';
            modeToggle.classList.add('draw-mode');
            drawControls.classList.add('visible');
        } else {
            modeToggle.textContent = 'Select Particle (T)';
            modeToggle.classList.remove('draw-mode');
            drawControls.classList.remove('visible');
            // Clear any active trail drawing
            system.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
        }
    }

    function toggleMode() {
        setMode(mouseMode === 'select' ? 'draw' : 'select');
    }

    modeToggle.addEventListener('click', toggleMode);

    // ─── Keyboard shortcuts ──────────────────────────────────────────────────

    window.addEventListener('keydown', (e) => {
        // Don't capture if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.key === 't' || e.key === 'T') {
            toggleMode();
        }
    });

    // ─── Mouse: coordinate conversion ────────────────────────────────────────

    function screenToWorld(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        const x = (clientX - rect.left) / rect.width * 2.0 - 1.0;
        const y = -((clientY - rect.top) / rect.height * 2.0 - 1.0);
        return { x, y };
    }

    function screenToUV(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        const u = (clientX - rect.left) / rect.width;
        const v = 1.0 - (clientY - rect.top) / rect.height; // flip Y for GL
        return { x: u, y: v };
    }

    // ─── Mouse: particle selection ───────────────────────────────────────────

    function selectParticleAt(clientX, clientY) {
        const world = screenToWorld(clientX, clientY);

        // Read entity positions from GPU
        const data = system.readEntityData();
        const c = system.c;

        // Find nearest entity
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

        // Compute the rule this entity was using
        const entityRule = computeEntityRule(
            config.rule,
            config.rule_seed,
            config.mutation_scale,
            config.cohorts,
            bestIndex,
            c.entityCount
        );

        // Push to history and apply
        ruleHistory.push(Array.from(entityRule), config.rule_seed);
        applyRule(entityRule, config.rule_seed);
    }

    // ─── Mouse events ────────────────────────────────────────────────────────

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // left click
            if (mouseMode === 'select') {
                selectParticleAt(e.clientX, e.clientY);
            } else if (mouseMode === 'draw') {
                mouseDown = true;
                const uv = screenToUV(e.clientX, e.clientY);
                mousePos = uv;
                prevMousePos = { ...uv }; // same as current on first click (no velocity)
            }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (mouseMode === 'draw' && mouseDown) {
            prevMousePos = { ...mousePos };
            mousePos = screenToUV(e.clientX, e.clientY);
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
            mouseDown = false;
            if (mouseMode === 'draw') {
                system.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
            }
        }
    });

    // Right click = undo
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        performUndo();
    });

    // ─── Config preview dropdown ─────────────────────────────────────────────

    function openDropdown() {
        if (dropdownOpen) return;
        dropdownOpen = true;
        presetDropdown.classList.add('open');
        updateActiveItem();
        // Save full config for restoration on cancel
        previewBaseConfig = { ...config, rule: Array.from(config.rule) };
    }

    function closeDropdown(restorePreview) {
        if (!dropdownOpen) return;
        dropdownOpen = false;
        presetDropdown.classList.remove('open');

        if (restorePreview && previewActive) {
            applyConfig(previewBaseConfig);
            previewActive = false;
        }
        previewBaseConfig = null;
    }

    // Click trigger to toggle dropdown
    presetTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        if (dropdownOpen) {
            closeDropdown(true);
        } else {
            openDropdown();
        }
    });

    // Hover on item → preview
    presetMenu.addEventListener('mouseover', async (e) => {
        const item = e.target.closest('.dropdown-item');
        if (!item || !dropdownOpen) return;

        const name = item.dataset.preset;
        const gen = ++previewGeneration; // track this hover event
        try {
            const previewConfig = await fetchConfig(name);
            // Stale if dropdown closed or a newer hover superseded this one
            if (!dropdownOpen || gen !== previewGeneration) return;

            // If we're already previewing, restore base first
            if (previewActive) {
                applyConfig(previewBaseConfig);
            }

            // Apply preview
            applyConfig(previewConfig);
            previewActive = true;
        } catch (err) {
            console.error(`Preview load failed: ${err.message}`);
        }
    });

    // Mouse leaves dropdown menu → restore
    presetMenu.addEventListener('mouseleave', () => {
        if (previewActive && dropdownOpen) {
            applyConfig(previewBaseConfig);
            previewActive = false;
        }
    });

    // Click on item → finalize selection
    presetMenu.addEventListener('click', async (e) => {
        const item = e.target.closest('.dropdown-item');
        if (!item) return;

        const name = item.dataset.preset;
        try {
            const newConfig = await fetchConfig(name);
            currentPresetName = name;
            presetTrigger.textContent = formatPresetName(name);

            // Don't restore preview — we're keeping this config
            previewActive = false;
            config = newConfig;
            system.setConfig(config);

            // Push to history
            ruleHistory.push(config.rule, config.rule_seed);

            closeDropdown(false);
        } catch (err) {
            showError(`Failed to load preset: ${err.message}`);
        }
    });

    // Click outside → close and restore
    document.addEventListener('click', (e) => {
        if (dropdownOpen && !presetDropdown.contains(e.target)) {
            closeDropdown(true);
        }
    });

    // Escape → close and restore
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && dropdownOpen) {
            closeDropdown(true);
        }
    });

    // ─── Debounced resize ────────────────────────────────────────────────────

    let resizeTimeout = null;
    window.addEventListener('resize', () => {
        resize();
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const ar = window.innerWidth / window.innerHeight;
            system.reinitGPU(worldSize, ar);
        }, 300);
    });

    // ─── Render loop ─────────────────────────────────────────────────────────

    function frame() {
        const physicsFrequency = parseInt(physicsFreqSlider.value, 10);

        // Update trail draw state each frame
        if (mouseMode === 'draw' && mouseDown) {
            system.setTrailDrawState({
                x: mousePos.x,
                y: mousePos.y,
                prevX: prevMousePos.x,
                prevY: prevMousePos.y,
                radius: parseFloat(drawSizeSlider.value),
                power: parseFloat(drawPowerSlider.value),
            });
        }

        for (let i = 0; i < physicsFrequency; i++) {
            system.advance();
        }

        // After advancing, sync prevMousePos so stationary mouse = zero velocity next frame
        if (mouseMode === 'draw' && mouseDown) {
            prevMousePos = { ...mousePos };
        }

        // Clear trail draw after advance so it only applies while mouse is held
        if (mouseMode === 'draw' && !mouseDown) {
            system.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
        }

        system.renderDisplay();
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);

    function showError(msg) {
        errorDisplay.textContent = msg;
        errorDisplay.style.display = 'block';
        console.error(msg);
    }
}

main();
