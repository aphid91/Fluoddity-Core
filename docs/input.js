/**
 * Input module: keyboard and mouse event binding.
 * Converts raw DOM events into action callbacks provided by the orchestrator.
 */

// ─── Coordinate conversion ──────────────────────────────────────────────────

export function screenToWorld(canvas, clientX, clientY, camera = null) {
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) / rect.width * 2.0 - 1.0;
    const y = -((clientY - rect.top) / rect.height * 2.0 - 1.0);
    if (camera) {
        return {
            x: x * camera.zoom + camera.posX,
            y: y * camera.zoom + camera.posY,
        };
    }
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
            case 'w': case 'W': state.cameraKeys.w = true; break;
            case 'a': case 'A': state.cameraKeys.a = true; break;
            case 's': case 'S': state.cameraKeys.s = true; break;
            case 'd': case 'D': state.cameraKeys.d = true; break;
            case 'q': case 'Q': state.cameraKeys.q = true; break;
            case 'e': case 'E': state.cameraKeys.e = true; break;
        }
    });

    window.addEventListener('keyup', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        switch (e.key) {
            case 'w': case 'W': state.cameraKeys.w = false; break;
            case 'a': case 'A': state.cameraKeys.a = false; break;
            case 's': case 'S': state.cameraKeys.s = false; break;
            case 'd': case 'D': state.cameraKeys.d = false; break;
            case 'q': case 'Q': state.cameraKeys.q = false; break;
            case 'e': case 'E': state.cameraKeys.e = false; break;
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

// ─── Scroll wheel zoom (Factorio-style: zoom toward mouse pointer) ──────────

export function setupScroll(canvas, state) {
    canvas.addEventListener('wheel', (e) => {
        if (!state.fancyCamera) return;
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const mouseNDC_x = (e.clientX - rect.left) / rect.width * 2.0 - 1.0;
        const mouseNDC_y = -((e.clientY - rect.top) / rect.height * 2.0 - 1.0);

        // World position under mouse with current camera
        const cam = state.camera;
        const worldX = mouseNDC_x * cam.zoom + cam.posX;
        const worldY = mouseNDC_y * cam.zoom + cam.posY;

        // Apply zoom
        const zoomFactor = e.deltaY > 0 ? 1.1 : 1.0 / 1.1;
        const newZoom = cam.zoom * zoomFactor;

        // Adjust camera so world point stays under mouse
        cam.posX = worldX - mouseNDC_x * newZoom;
        cam.posY = worldY - mouseNDC_y * newZoom;
        cam.zoom = Math.max(0.01, Math.min(100.0, newZoom));
    }, { passive: false });
}

// ─── Continuous camera update (called once per frame) ───────────────────────

export function updateCamera(state, dt) {
    if (!state.fancyCamera) return;
    const keys = state.cameraKeys;
    const cam = state.camera;

    const panSpeed = 1.5 * cam.zoom * dt;
    if (keys.a) cam.posX -= panSpeed;
    if (keys.d) cam.posX += panSpeed;
    if (keys.w) cam.posY += panSpeed;
    if (keys.s) cam.posY -= panSpeed;

    const zoomSpeed = 1.5 * dt;
    if (keys.q) cam.zoom *= (1.0 + zoomSpeed);
    if (keys.e) cam.zoom *= (1.0 - zoomSpeed);
    cam.zoom = Math.max(0.01, Math.min(100.0, cam.zoom));
}
