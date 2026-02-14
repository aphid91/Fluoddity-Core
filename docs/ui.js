/**
 * UI module: DOM element setup, dropdown behavior, slider wiring.
 * Returns references to UI elements so the orchestrator can read slider values.
 */

import { fetchConfig } from './state.js';

// ─── UI element references ──────────────────────────────────────────────────

/**
 * Grab all UI element references from the DOM.
 * @returns {object} elements map
 */
function getElements() {
    return {
        worldSizeSelector: document.getElementById('world-size-selector'),
        resetButton: document.getElementById('reset-button'),
        undoButton: document.getElementById('undo-button'),
        physicsFreqSlider: document.getElementById('physics-freq'),
        physicsFreqValue: document.getElementById('physics-freq-value'),
        modeToggle: document.getElementById('mode-toggle'),
        drawControls: document.getElementById('draw-controls'),
        drawSizeSlider: document.getElementById('draw-size'),
        drawSizeValue: document.getElementById('draw-size-value'),
        drawPowerSlider: document.getElementById('draw-power'),
        drawPowerValue: document.getElementById('draw-power-value'),
        presetDropdown: document.getElementById('preset-dropdown'),
        presetTrigger: document.getElementById('preset-trigger'),
        presetMenu: document.getElementById('preset-menu'),
        brightnessSlider: document.getElementById('brightness-slider'),
        brightnessValue: document.getElementById('brightness-value'),
        initialConditionsToggle: document.getElementById('initial-conditions-toggle'),
        mutationScaleSlider: document.getElementById('mutation-scale-slider'),
        mutationScaleValue: document.getElementById('mutation-scale-value'),
        cohortsSelector: document.getElementById('cohorts-selector'),
    };
}

// ─── Slider setup (with auto-blur fix) ──────────────────────────────────────

function setupSliders(el, actions) {
    el.physicsFreqSlider.addEventListener('input', () => {
        el.physicsFreqValue.textContent = el.physicsFreqSlider.value;
    });
    el.physicsFreqSlider.addEventListener('change', () => el.physicsFreqSlider.blur());

    el.drawSizeSlider.addEventListener('input', () => {
        el.drawSizeValue.textContent = parseFloat(el.drawSizeSlider.value).toFixed(3);
    });
    el.drawSizeSlider.addEventListener('change', () => el.drawSizeSlider.blur());

    el.drawPowerSlider.addEventListener('input', () => {
        el.drawPowerValue.textContent = parseFloat(el.drawPowerSlider.value).toFixed(2);
    });
    el.drawPowerSlider.addEventListener('change', () => el.drawPowerSlider.blur());

    el.brightnessSlider.addEventListener('input', () => {
        el.brightnessValue.textContent = parseFloat(el.brightnessSlider.value).toFixed(3);
    });
    el.brightnessSlider.addEventListener('change', () => el.brightnessSlider.blur());

    el.mutationScaleSlider.addEventListener('input', () => {
        el.mutationScaleValue.textContent = parseFloat(el.mutationScaleSlider.value).toFixed(3);
        actions.onMutationScaleChange(parseFloat(el.mutationScaleSlider.value));
    });
    el.mutationScaleSlider.addEventListener('change', () => el.mutationScaleSlider.blur());
}

// ─── Mode toggle ────────────────────────────────────────────────────────────

function setupModeToggle(el, state, actions) {
    el.modeToggle.addEventListener('click', actions.toggleMode);
}

export function updateModeDisplay(el, state) {
    if (state.mouseMode === 'draw') {
        el.modeToggle.textContent = 'Draw Trail (T)';
        el.modeToggle.title = 'Mouse mode: In Draw Trail mode, click and drag to influence particles';
        el.modeToggle.classList.add('draw-mode');
        el.drawControls.classList.add('visible');
    } else {
        el.modeToggle.textContent = 'Select Particle (T)';
        el.modeToggle.title = "Mouse mode: In Select Particle mode, click on a particle to adopt it's behavior (with mutations)";
        el.modeToggle.classList.remove('draw-mode');
        el.drawControls.classList.remove('visible');
    }
}

// ─── Initial conditions display ─────────────────────────────────────────────

const INITIAL_CONDITIONS_LABELS = ['Grid', 'Random', 'Ring'];

export function updateInitialConditionsDisplay(el, config) {
    const mode = config.initial_conditions || 0;
    const label = INITIAL_CONDITIONS_LABELS[mode] || `Mode ${mode}`;
    el.initialConditionsToggle.textContent = `Starting Positions: ${label}`;
}

// ─── Mutation scale display ─────────────────────────────────────────────────

export function updateMutationScaleDisplay(el, config) {
    const val = config.mutation_scale;
    if (val > parseFloat(el.mutationScaleSlider.max)) {
        el.mutationScaleSlider.max = String(val);
    }
    if (val < parseFloat(el.mutationScaleSlider.min)) {
        el.mutationScaleSlider.min = String(val);
    }
    el.mutationScaleSlider.value = String(val);
    el.mutationScaleValue.textContent = val.toFixed(3);
}

// ─── Cohorts display ────────────────────────────────────────────────────────

export function updateCohortsDisplay(el, config) {
    const val = config.cohorts;
    const opts = el.cohortsSelector.options;
    // Remove any previously added non-standard option
    for (let i = opts.length - 1; i >= 0; i--) {
        if (opts[i].dataset.nonstandard === 'true') {
            el.cohortsSelector.remove(i);
        }
    }
    let found = false;
    for (let i = 0; i < opts.length; i++) {
        if (parseInt(opts[i].value) === val) {
            el.cohortsSelector.selectedIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        const opt = document.createElement('option');
        opt.value = String(val);
        opt.textContent = String(val);
        opt.dataset.nonstandard = 'true';
        el.cohortsSelector.appendChild(opt);
        el.cohortsSelector.value = String(val);
    }
}

// ─── Preset dropdown ────────────────────────────────────────────────────────

function formatPresetName(name) {
    return name.replace(/([A-Z])/g, ' $1').trim();
}

function setupDropdown(el, presetNames, state, actions) {
    // Populate dropdown items
    for (const name of presetNames) {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.dataset.preset = name;
        item.textContent = formatPresetName(name);
        el.presetMenu.appendChild(item);
    }

    function updateActiveItem() {
        for (const item of el.presetMenu.children) {
            item.classList.toggle('active', item.dataset.preset === actions.getCurrentPresetName());
        }
    }

    function openDropdown() {
        if (state.dropdownOpen) return;
        state.dropdownOpen = true;
        el.presetDropdown.classList.add('open');
        updateActiveItem();
        state.previewBaseConfig = actions.snapshotConfig();
    }

    function closeDropdown(restorePreview) {
        if (!state.dropdownOpen) return;
        state.dropdownOpen = false;
        el.presetDropdown.classList.remove('open');

        if (restorePreview && state.previewActive) {
            actions.applyConfig(state.previewBaseConfig);
            state.previewActive = false;
        }
        state.previewBaseConfig = null;
    }

    // Expose closeDropdown for keyboard handler
    actions.registerCloseDropdown(closeDropdown);

    // Click trigger to toggle dropdown
    el.presetTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        if (state.dropdownOpen) {
            closeDropdown(true);
        } else {
            openDropdown();
        }
    });

    // Hover on item -> preview
    el.presetMenu.addEventListener('mouseover', async (e) => {
        const item = e.target.closest('.dropdown-item');
        if (!item || !state.dropdownOpen) return;

        const name = item.dataset.preset;
        const gen = ++state.previewGeneration;
        try {
            const previewConfig = await fetchConfig(name);
            if (!state.dropdownOpen || gen !== state.previewGeneration) return;

            if (state.previewActive) {
                actions.applyConfig(state.previewBaseConfig);
            }

            actions.applyConfig(previewConfig);
            state.previewActive = true;
        } catch (err) {
            console.error(`Preview load failed: ${err.message}`);
        }
    });

    // Mouse leaves dropdown menu -> restore
    el.presetMenu.addEventListener('mouseleave', () => {
        if (state.previewActive && state.dropdownOpen) {
            actions.applyConfig(state.previewBaseConfig);
            state.previewActive = false;
        }
    });

    // Click on item -> finalize selection
    el.presetMenu.addEventListener('click', async (e) => {
        const item = e.target.closest('.dropdown-item');
        if (!item) return;

        const name = item.dataset.preset;
        try {
            const newConfig = await fetchConfig(name);
            actions.setCurrentPresetName(name);
            el.presetTrigger.textContent = formatPresetName(name);

            state.previewActive = false;
            actions.finalizePresetLoad(newConfig);

            closeDropdown(false);
        } catch (err) {
            actions.logError(`Failed to load preset: ${err.message}`);
        }
    });

    // Click outside -> close and restore
    document.addEventListener('click', (e) => {
        if (state.dropdownOpen && !el.presetDropdown.contains(e.target)) {
            closeDropdown(true);
        }
    });

    return { updateActiveItem, closeDropdown };
}

// ─── Main setup ─────────────────────────────────────────────────────────────

/**
 * Initialize all UI elements, wire up events.
 * @param {string[]} presetNames - preset manifest
 * @param {string} initialPresetName - first preset to display
 * @param {object} state - shared AppState
 * @param {object} actions - callbacks from orchestrator
 * @returns {object} { el, dropdown }
 */
export function setupUI(presetNames, initialPresetName, state, actions) {
    const el = getElements();

    el.presetTrigger.textContent = formatPresetName(initialPresetName);

    setupSliders(el, actions);
    setupModeToggle(el, state, actions);

    el.resetButton.addEventListener('click', actions.resetSimulation);
    el.undoButton.addEventListener('click', actions.performUndo);

    el.worldSizeSelector.addEventListener('change', () => {
        actions.changeWorldSize(parseFloat(el.worldSizeSelector.value));
        el.worldSizeSelector.blur();
    });

    el.initialConditionsToggle.addEventListener('click', () => {
        actions.cycleInitialConditions();
        el.initialConditionsToggle.blur();
    });

    el.cohortsSelector.addEventListener('change', () => {
        actions.onCohortsChange(parseInt(el.cohortsSelector.value));
        el.cohortsSelector.blur();
    });

    const dropdown = setupDropdown(el, presetNames, state, actions);

    return { el, dropdown };
}
