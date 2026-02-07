/**
 * Main entry point for Fluoddity WebGL 2 port.
 * Sets up canvas, WebGL context, loads config, and runs the render loop.
 */

import { loadConfig } from './gl_utils.js';
import { ParticleSystem } from './particle_system.js';

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

    // Load preset manifest and populate dropdown
    const presetSelector = document.getElementById('preset-selector');
    let presetNames;
    try {
        const manifestResponse = await fetch('physics_configs/index.json');
        if (!manifestResponse.ok) throw new Error(`HTTP ${manifestResponse.status}`);
        presetNames = await manifestResponse.json();
    } catch (e) {
        showError(`Failed to load preset manifest: ${e.message}`);
        return;
    }

    // Populate dropdown from manifest
    presetSelector.innerHTML = '';
    for (const name of presetNames) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name.replace(/([A-Z])/g, ' $1').trim();
        presetSelector.appendChild(option);
    }

    // Load default config
    let config;
    try {
        config = await loadPreset(presetSelector.value);
    } catch (e) {
        showError(`Failed to load config: ${e.message}`);
        return;
    }

    // World size selector
    const worldSizeSelector = document.getElementById('world-size-selector');
    let worldSize = parseFloat(worldSizeSelector.value);

    // Create and initialize particle system
    const aspectRatio = window.innerWidth / window.innerHeight;
    const system = new ParticleSystem(gl, config, worldSize, aspectRatio);
    try {
        await system.init();
    } catch (e) {
        showError(`Initialization failed: ${e.message}`);
        console.error(e);
        return;
    }

    // World size change → full GPU reinit
    worldSizeSelector.addEventListener('change', () => {
        worldSize = parseFloat(worldSizeSelector.value);
        const ar = window.innerWidth / window.innerHeight;
        system.reinitGPU(worldSize, ar);
    });

    // Preset selector — changes config without resetting particles
    presetSelector.addEventListener('change', async () => {
        try {
            const newConfig = await loadPreset(presetSelector.value);
            system.setConfig(newConfig);
        } catch (e) {
            showError(`Failed to load preset: ${e.message}`);
        }
    });

    // Reset button
    const resetButton = document.getElementById('reset-button');
    resetButton.addEventListener('click', () => {
        system.reset();
    });

    // Debounced resize → reinit with new aspect ratio
    let resizeTimeout = null;
    window.addEventListener('resize', () => {
        resize();
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const ar = window.innerWidth / window.innerHeight;
            system.reinitGPU(worldSize, ar);
        }, 300);
    });

    // Render loop
    function frame() {
        // 5 simulation steps per display frame (180Hz sim at 60Hz display)
        for (let i = 0; i < 5; i++) {
            system.advance();
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

async function loadPreset(name) {
    const path = `physics_configs/${name}.json`;
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${path}`);
    const data = await response.json();
    return loadConfig(data);
}

main();
