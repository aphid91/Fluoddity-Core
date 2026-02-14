/**
 * Input module: keyboard and mouse event binding.
 * Converts raw DOM events into action callbacks provided by the orchestrator.
 */

// ─── Coordinate conversion ──────────────────────────────────────────────────

export function screenToWorld(canvas, clientX, clientY, camera = null, constants = null) {
    const rect = canvas.getBoundingClientRect();
    const ndcX = (clientX - rect.left) / rect.width * 2.0 - 1.0;
    const ndcY = -((clientY - rect.top) / rect.height * 2.0 - 1.0);
    if (camera && constants) {
        // Invert the cam_brush.vert transform:
        //   ndc = vertex_pos * vec2(1, tex_aspect) * scale - cam_pos * vec2(1,-1) / cam_zoom
        // where scale = base_scale / cam_zoom
        // So: ndc * cam_zoom = vertex_pos * vec2(1, tex_aspect) * base_scale - cam_pos * vec2(1,-1)
        // Solving for vertex_pos (entity world position):
        const texAspect = constants.canvasWidth / constants.canvasHeight;
        const winAspect = canvas.width / canvas.height;
        let bsx, bsy;
        if (texAspect > winAspect) {
            bsx = 1.0;
            bsy = winAspect / texAspect;
        } else {
            bsx = texAspect / winAspect;
            bsy = 1.0;
        }
        const worldX = (ndcX * camera.zoom + camera.posX) / bsx;
        const worldY = (ndcY * camera.zoom - camera.posY) / (texAspect * bsy);
        return { x: worldX, y: worldY };
    }
    return { x: ndcX, y: ndcY };
}

export function screenToUV(canvas, clientX, clientY, camera = null, constants = null) {
    const rect = canvas.getBoundingClientRect();
    if (camera && constants) {
        // Convert screen position through camera to canvas UV
        const world = screenToWorld(canvas, clientX, clientY, camera, constants);
        // Entity world space [-1,1] → canvas UV [0,1]
        const u = (world.x + 1.0) * 0.5;
        const v = (world.y + 1.0) * 0.5;
        return { x: u, y: v };
    }
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
            case 'c':
            case 'C':
                actions.saveConfig();
                break;
            case 'v':
            case 'V':
                actions.loadConfig();
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
 *   { selectParticleAt, performUndo, setTrailDrawState, getConstants }
 */
export function setupMouse(canvas, state, actions) {
    function getCameraUV(clientX, clientY) {
        const cam = state.fancyCamera ? state.camera : null;
        const c = cam ? actions.getConstants() : null;
        return screenToUV(canvas, clientX, clientY, cam, c);
    }

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0) {
            if (state.mouseMode === 'select') {
                actions.selectParticleAt(e.clientX, e.clientY);
            } else if (state.mouseMode === 'draw') {
                state.mouseDown = true;
                const uv = getCameraUV(e.clientX, e.clientY);
                state.mousePos = uv;
                state.prevMousePos = { ...uv };
            }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (state.mouseMode === 'draw' && state.mouseDown) {
            state.prevMousePos = { ...state.mousePos };
            state.mousePos = getCameraUV(e.clientX, e.clientY);
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

        // Shader: ndc.x * zoom = world.x * base_sx - posX
        //         ndc.y * zoom = world.y * base_sy + posY
        // For zoom-to-cursor (keep ndc fixed while zoom changes):
        //   posX_new = posX_old - ndc.x * (zoom_new - zoom_old)
        //   posY_new = posY_old + ndc.y * (zoom_new - zoom_old)
        const cam = state.camera;
        const zoomFactor = e.deltaY > 0 ? 1.1 : 1.0 / 1.1;
        const newZoom = cam.zoom * zoomFactor;
        const dZoom = newZoom - cam.zoom;

        cam.posX -= mouseNDC_x * dZoom;
        cam.posY += mouseNDC_y * dZoom;
        cam.zoom = Math.max(0.01, Math.min(100.0, newZoom));
    }, { passive: false });
}

// ─── Continuous camera update (called once per frame) ───────────────────────

export function updateCamera(state, dt) {
    if (!state.fancyCamera) return;
    const keys = state.cameraKeys;
    const cam = state.camera;

    // Shader uses: ndc -= cam_pos * vec2(1, -1) / cam_zoom
    // So increasing posX shifts view right, decreasing posY shifts view up
    const panSpeed = 1.5 * cam.zoom * dt;
    if (keys.a) cam.posX -= panSpeed;
    if (keys.d) cam.posX += panSpeed;
    if (keys.w) cam.posY -= panSpeed;
    if (keys.s) cam.posY += panSpeed;

    const zoomSpeed = 1.5 * dt;
    if (keys.q) cam.zoom *= (1.0 + zoomSpeed);
    if (keys.e) cam.zoom *= (1.0 - zoomSpeed);
    cam.zoom = Math.max(0.01, Math.min(100.0, cam.zoom));
}
