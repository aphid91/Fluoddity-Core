/**
 * ParticleSystem: orchestrates the 3-stage GPGPU simulation pipeline.
 * Mirrors particle_system.py from the desktop version.
 *
 * Pipeline per advance():
 *   1. Entity Update  (fullscreen quad GPGPU → entity texture)
 *   2. Brush Creation  (instanced draw → brush texture)
 *   3. Canvas Update   (fullscreen quad → canvas texture)
 */

import {
    createProgram, createFloatTexture, createFramebuffer,
    tryset, setConfigUniforms, setRuleUniforms, fetchShader
} from './gl_utils.js';

// Derive all simulation constants from worldSize
function computeConstants(worldSize, aspectRatio) {
    const sqrtWorldSize = Math.sqrt(worldSize);
    const entityCount = Math.floor(600000 * worldSize);
    const canvasDim = Math.floor(1024 * sqrtWorldSize);
    const sqrtAspect = Math.sqrt(aspectRatio);
    const canvasWidth = Math.floor(canvasDim * sqrtAspect);
    const canvasHeight = Math.floor(canvasDim / sqrtAspect);
    const entityTexWidth = Math.ceil(Math.sqrt(entityCount));
    const entityTexHeight = Math.ceil(entityCount / entityTexWidth);
    return { worldSize, sqrtWorldSize, entityCount, canvasDim, canvasWidth, canvasHeight, entityTexWidth, entityTexHeight };
}

export class ParticleSystem {
    constructor(gl, config, worldSize, aspectRatio) {
        this.gl = gl;
        this.config = config;
        this.frameCount = 0;
        this.c = computeConstants(worldSize, aspectRatio);

        // Trail drawing state: current + previous mouse position, radius, power
        this.trailDrawState = { x: 0, y: 0, prevX: 0, prevY: 0, radius: 0, power: 0 };

        // Programs (set in init(), persist across reinitGPU)
        this.entityUpdateProgram = null;
        this.brushProgram = null;
        this.canvasUpdateProgram = null;
        this.cameraProgram = null;

        // GPU resources (set in _createGPUResources)
        this.entityTextures = [null, null];
        this.entityFBOs = [null, null];
        this.entityPing = 0;

        this.brushTexture = null;
        this.brushFBO = null;

        this.canvasTextures = [null, null];
        this.canvasFBOs = [null, null];
        this.canvasPing = 0;

        this.fullscreenQuadVAO = null;
        this.canvasQuadVAO = null;
        this.brushVAO = null;
        this.cameraVAO = null;
    }

    async init() {
        const gl = this.gl;

        // Load all shaders
        const [
            fullscreenQuadVert,
            entityUpdateFrag,
            brushVert,
            brushFrag,
            canvasFrag,
            cameraFrag
        ] = await Promise.all([
            fetchShader('shaders/fullscreen_quad.vert'),
            fetchShader('shaders/entity_update.frag'),
            fetchShader('shaders/brush.vert'),
            fetchShader('shaders/brush.frag'),
            fetchShader('shaders/canvas.frag'),
            fetchShader('shaders/camera.frag'),
        ]);

        // Compile programs
        this.entityUpdateProgram = createProgram(gl, fullscreenQuadVert, entityUpdateFrag);
        this.brushProgram = createProgram(gl, brushVert, brushFrag);
        this.canvasUpdateProgram = createProgram(gl, fullscreenQuadVert, canvasFrag);
        this.cameraProgram = createProgram(gl, fullscreenQuadVert, cameraFrag);

        // Create GPU resources
        this._createGPUResources();
    }

    reinitGPU(worldSize, aspectRatio) {
        this.c = computeConstants(worldSize, aspectRatio);
        this._destroyGPUResources();
        this._createGPUResources();
        this.frameCount = 0;
    }

    _createGPUResources() {
        const gl = this.gl;
        const c = this.c;

        // Entity textures (ping-pong)
        for (let i = 0; i < 2; i++) {
            this.entityTextures[i] = createFloatTexture(gl, c.entityTexWidth, c.entityTexHeight);
            this.entityFBOs[i] = createFramebuffer(gl, this.entityTextures[i]);
        }
        this.entityPing = 0;

        // Brush texture
        this.brushTexture = createFloatTexture(gl, c.canvasWidth, c.canvasHeight);
        this.brushFBO = createFramebuffer(gl, this.brushTexture);

        // Canvas textures (ping-pong)
        for (let i = 0; i < 2; i++) {
            this.canvasTextures[i] = createFloatTexture(gl, c.canvasWidth, c.canvasHeight);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            this.canvasFBOs[i] = createFramebuffer(gl, this.canvasTextures[i]);
        }
        this.canvasPing = 0;

        // VAOs
        this._createFullscreenQuadVAOs();
        this._createBrushVAO();
    }

    _destroyGPUResources() {
        const gl = this.gl;

        for (let i = 0; i < 2; i++) {
            if (this.entityTextures[i]) gl.deleteTexture(this.entityTextures[i]);
            if (this.entityFBOs[i]) gl.deleteFramebuffer(this.entityFBOs[i]);
            if (this.canvasTextures[i]) gl.deleteTexture(this.canvasTextures[i]);
            if (this.canvasFBOs[i]) gl.deleteFramebuffer(this.canvasFBOs[i]);
        }
        if (this.brushTexture) gl.deleteTexture(this.brushTexture);
        if (this.brushFBO) gl.deleteFramebuffer(this.brushFBO);

        if (this.fullscreenQuadVAO) gl.deleteVertexArray(this.fullscreenQuadVAO);
        if (this.canvasQuadVAO) gl.deleteVertexArray(this.canvasQuadVAO);
        if (this.cameraVAO) gl.deleteVertexArray(this.cameraVAO);
        if (this.brushVAO) gl.deleteVertexArray(this.brushVAO);
    }

    _createFullscreenQuadVAOs() {
        const vertices = new Float32Array([
            -1, -1,  1, -1,  1,  1,
            -1, -1,  1,  1, -1,  1,
        ]);
        this.fullscreenQuadVAO = this._makeQuadVAO(this.entityUpdateProgram, vertices);
        this.canvasQuadVAO = this._makeQuadVAO(this.canvasUpdateProgram, vertices);
        this.cameraVAO = this._makeQuadVAO(this.cameraProgram, vertices);
    }

    _makeQuadVAO(program, vertices) {
        const gl = this.gl;
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        const loc = gl.getAttribLocation(program, 'in_position');
        if (loc >= 0) {
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        }

        gl.bindVertexArray(null);
        return vao;
    }

    _createBrushVAO() {
        const gl = this.gl;

        const quadData = new Float32Array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ]);

        this.brushVAO = gl.createVertexArray();
        gl.bindVertexArray(this.brushVAO);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, quadData, gl.STATIC_DRAW);

        const offsetLoc = gl.getAttribLocation(this.brushProgram, 'a_offset');
        const uvLoc = gl.getAttribLocation(this.brushProgram, 'a_uv');

        if (offsetLoc >= 0) {
            gl.enableVertexAttribArray(offsetLoc);
            gl.vertexAttribPointer(offsetLoc, 2, gl.FLOAT, false, 16, 0);
        }
        if (uvLoc >= 0) {
            gl.enableVertexAttribArray(uvLoc);
            gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
        }

        gl.bindVertexArray(null);
    }

    advance() {
        this.createBrush();
        this.updateEntities();
        this.updateCanvas();
        this.frameCount++;
    }

    updateEntities() {
        const gl = this.gl;
        const prog = this.entityUpdateProgram;
        const c = this.c;
        const readIdx = this.entityPing;
        const writeIdx = 1 - readIdx;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.entityFBOs[writeIdx]);
        gl.viewport(0, 0, c.entityTexWidth, c.entityTexHeight);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.entityTextures[readIdx]);
        tryset(gl, prog, 'entity_texture', 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.canvasTextures[this.canvasPing]);
        tryset(gl, prog, 'canvas_texture', 1);

        setConfigUniforms(gl, prog, this.config);
        setRuleUniforms(gl, prog, this.config.rule);
        tryset(gl, prog, 'frame_count', this.frameCount);
        tryset(gl, prog, 'entity_count', c.entityCount);
        tryset(gl, prog, 'entity_tex_width', c.entityTexWidth);
        tryset(gl, prog, 'sqrt_world_size', c.sqrtWorldSize, 'float');

        gl.bindVertexArray(this.fullscreenQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        this.entityPing = writeIdx;
    }

    createBrush() {
        const gl = this.gl;
        const prog = this.brushProgram;
        const c = this.c;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.brushFBO);
        gl.viewport(0, 0, c.canvasWidth, c.canvasHeight);

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.entityTextures[this.entityPing]);
        tryset(gl, prog, 'entity_texture', 0);

        tryset(gl, prog, 'canvas_resolution', [c.canvasWidth, c.canvasHeight]);
        tryset(gl, prog, 'frame_count', this.frameCount);
        tryset(gl, prog, 'entity_tex_width', c.entityTexWidth);
        tryset(gl, prog, 'sqrt_world_size', c.sqrtWorldSize, 'float');

        gl.bindVertexArray(this.brushVAO);
        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, c.entityCount);

        gl.disable(gl.BLEND);
    }

    updateCanvas() {
        const gl = this.gl;
        const prog = this.canvasUpdateProgram;
        const c = this.c;
        const readIdx = this.canvasPing;
        const writeIdx = 1 - readIdx;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.canvasFBOs[writeIdx]);
        gl.viewport(0, 0, c.canvasWidth, c.canvasHeight);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
        tryset(gl, prog, 'brush_texture', 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.canvasTextures[readIdx]);
        tryset(gl, prog, 'canvas_texture', 1);

        setConfigUniforms(gl, prog, this.config);
        tryset(gl, prog, 'frame_count', this.frameCount);

        // Trail drawing uniforms
        const td = this.trailDrawState;
        tryset(gl, prog, 'trail_draw_mouse', [td.x, td.y, td.prevX, td.prevY]);
        tryset(gl, prog, 'trail_draw_radius', td.radius, 'float');
        tryset(gl, prog, 'trail_draw_power', td.power, 'float');

        gl.bindVertexArray(this.canvasQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        this.canvasPing = writeIdx;
    }

    renderDisplay() {
        const gl = this.gl;
        const prog = this.cameraProgram;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
        tryset(gl, prog, 'tex', 0);

        gl.bindVertexArray(this.cameraVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    reset() {
        this.frameCount = 0;
    }

    setConfig(config) {
        this.config = config;
    }

    setTrailDrawState(state) {
        this.trailDrawState = state;
    }

    /**
     * Read back the entity texture from GPU.
     * Returns a Float32Array of (entityTexWidth * entityTexHeight * 4) floats.
     * Each entity occupies 4 consecutive floats: [pos.x, pos.y, vel.x, vel.y].
     */
    readEntityData() {
        const gl = this.gl;
        const c = this.c;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.entityFBOs[this.entityPing]);
        const data = new Float32Array(c.entityTexWidth * c.entityTexHeight * 4);
        gl.readPixels(0, 0, c.entityTexWidth, c.entityTexHeight, gl.RGBA, gl.FLOAT, data);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return data;
    }
}
