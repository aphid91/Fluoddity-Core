/**
 * Main entry point for Fluoddity WebGL 2 port.
 * Sets up canvas, WebGL context, loads config, and runs the render loop.
 */

import { loadConfig } from './gl_utils.js';
import { ParticleSystem } from './particle_system.js';

const PRESETS = {
    'HungryHungryHippos': 'configs/HungryHungryHippos.json',
    'Searching': 'configs/Searching.json',
};

async function main() {
    const canvas = document.getElementById('canvas');
    const errorDisplay = document.getElementById('error-display');

    // Size canvas to window
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

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

    // Load default config
    const presetSelector = document.getElementById('preset-selector');
    let config;
    try {
        config = await loadPreset(presetSelector.value);
    } catch (e) {
        showError(`Failed to load config: ${e.message}`);
        return;
    }

    // Create and initialize particle system
    const system = new ParticleSystem(gl, config);
    try {
        await system.init();
    } catch (e) {
        showError(`Initialization failed: ${e.message}`);
        console.error(e);
        return;
    }

    // Preset selector handler
    presetSelector.addEventListener('change', async () => {
        try {
            const newConfig = await loadPreset(presetSelector.value);
            system.setConfig(newConfig);
        } catch (e) {
            showError(`Failed to load preset: ${e.message}`);
        }
    });

    // Render loop
    function frame() {
        // Resize viewport if canvas changed
        if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
            resize();
        }

        // 5 simulation steps per display frame (180Hz sim at 60Hz display)
        for (let i = 0; i < 5; i++) {
            system.advance();
        }

        // Display
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
    const path = PRESETS[name];
    if (!path) throw new Error(`Unknown preset: ${name}`);
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${path}`);
    const data = await response.json();
    return loadConfig(data);
}

main();
