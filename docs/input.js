/**
 * Input module: keyboard and mouse event binding.
 * Converts raw DOM events into action callbacks provided by the orchestrator.
 */

// ─── Coordinate conversion ──────────────────────────────────────────────────

export function screenToWorld(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) / rect.width * 2.0 - 1.0;
    const y = -((clientY - rect.top) / rect.height * 2.0 - 1.0);
    return { x, y };
}

export function screenToUV(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const u = (clientX - rect.left) / rect.width;
    const v = 1.0 - (clientY - rect.top) / rect.height;
    return { x: u, y: v };
}

// ─── Keyboard ───────────────────────────────────────────────────────────────

/**
 * @param {object} state - shared AppState
 * @param {object} actions - callbacks:
 *   { toggleMode, performUndo, resetSimulation, randomizeSeed, closeDropdown }
 */
export function setupKeyboard(state, actions) {
    window.addEventListener('keydown', (e) => {
        // Don't capture if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.key) {
            case 't':
            case 'T':
                actions.toggleMode();
                break;
            case ' ':
                e.preventDefault();
                actions.togglePause();
                break;
            case 'r':
            case 'R':
                actions.resetSimulation();
                break;
            case 'g':
            case 'G':
                actions.randomizeSeed();
                break;
            case 'Escape':
                if (state.dropdownOpen) {
                    actions.closeDropdown(true);
                }
                break;
        }
    });
}

// ─── Mouse ──────────────────────────────────────────────────────────────────

/**
 * @param {HTMLCanvasElement} canvas
 * @param {object} state - shared AppState
 * @param {object} actions - callbacks:
 *   { selectParticleAt, performUndo, setTrailDrawState }
 */
export function setupMouse(canvas, state, actions) {
    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0) {
            if (state.mouseMode === 'select') {
                actions.selectParticleAt(e.clientX, e.clientY);
            } else if (state.mouseMode === 'draw') {
                state.mouseDown = true;
                const uv = screenToUV(canvas, e.clientX, e.clientY);
                state.mousePos = uv;
                state.prevMousePos = { ...uv };
            }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (state.mouseMode === 'draw' && state.mouseDown) {
            state.prevMousePos = { ...state.mousePos };
            state.mousePos = screenToUV(canvas, e.clientX, e.clientY);
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
            state.mouseDown = false;
            if (state.mouseMode === 'draw') {
                actions.setTrailDrawState({ x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 });
            }
        }
    });

    // Right click = undo
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        actions.performUndo();
    });
}
