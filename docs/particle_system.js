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
        this.camBrushProgram = null;

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
        this.camBrushVAO = null;

        // Bloom/tonemap resources (lazy init on first fancy camera render)
        this._bloomInitialized = false;
        this._bloomInitializing = false;
        this._bloomWidth = 0;
        this._bloomHeight = 0;
        this._fullscreenQuadVertSrc = null;

        this.bloomDownsampleProgram = null;
        this.bloomUpsampleProgram = null;
        this.tonemapProgram = null;

        this.camBrushTexture = null;
        this.camBrushFBO = null;
        this.bloomCopyTexture = null;
        this.bloomCopyFBO = null;
        this.bloomMipTextures = [];
        this.bloomMipFBOs = [];
        this.bloomDownsampleVAO = null;
        this.bloomUpsampleVAO = null;
        this.tonemapVAO = null;
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
            cameraFrag,
            camBrushVert,
            camBrushFrag
        ] = await Promise.all([
            fetchShader('shaders/fullscreen_quad.vert'),
            fetchShader('shaders/entity_update.frag'),
            fetchShader('shaders/brush.vert'),
            fetchShader('shaders/brush.frag'),
            fetchShader('shaders/canvas.frag'),
            fetchShader('shaders/camera.frag'),
            fetchShader('shaders/cam_brush.vert'),
            fetchShader('shaders/cam_brush.frag'),
        ]);

        // Cache vertex shader source for lazy bloom init
        this._fullscreenQuadVertSrc = fullscreenQuadVert;

        // Compile programs
        this.entityUpdateProgram = createProgram(gl, fullscreenQuadVert, entityUpdateFrag);
        this.brushProgram = createProgram(gl, brushVert, brushFrag);
        this.canvasUpdateProgram = createProgram(gl, fullscreenQuadVert, canvasFrag);
        this.cameraProgram = createProgram(gl, fullscreenQuadVert, cameraFrag);
        this.camBrushProgram = createProgram(gl, camBrushVert, camBrushFrag);

        // Create GPU resources
        this._createGPUResources();
    }

    reinitGPU(worldSize, aspectRatio) {
        this.c = computeConstants(worldSize, aspectRatio);
        this._destroyGPUResources();
        this._createGPUResources();
        this.frameCount = 0;

        // Recreate bloom FBOs at new canvas size if bloom was initialized
        if (this._bloomInitialized) {
            const gl = this.gl;
            this._createBloomFBOs(gl.canvas.width, gl.canvas.height);
            this._createBloomVAOs();
        }
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
        this._createCamBrushVAO();
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
        if (this.camBrushVAO) gl.deleteVertexArray(this.camBrushVAO);

        // Bloom FBOs/VAOs are size-dependent; programs are kept
        if (this._bloomInitialized) {
            this._destroyBloomFBOs();
            this._destroyBloomVAOs();
        }
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

    _createCamBrushVAO() {
        const gl = this.gl;
        // Empty VAO - cam_brush.vert uses gl_VertexID to index local arrays
        this.camBrushVAO = gl.createVertexArray();
    }

    // ─── Bloom/tonemap lazy initialization ────────────────────────────────────

    async _initBloomResources() {
        const gl = this.gl;

        // Fetch and compile bloom/tonemap shaders
        const [downsampleFrag, upsampleFrag, tonemapFrag] = await Promise.all([
            fetchShader('shaders/bloom_downsample.frag'),
            fetchShader('shaders/bloom_upsample.frag'),
            fetchShader('shaders/tonemap.frag'),
        ]);

        this.bloomDownsampleProgram = createProgram(gl, this._fullscreenQuadVertSrc, downsampleFrag);
        this.bloomUpsampleProgram = createProgram(gl, this._fullscreenQuadVertSrc, upsampleFrag);
        this.tonemapProgram = createProgram(gl, this._fullscreenQuadVertSrc, tonemapFrag);

        // Create FBOs at current screen size
        this._createBloomFBOs(gl.canvas.width, gl.canvas.height);

        // Create VAOs for postprocess passes
        this._createBloomVAOs();

        this._bloomInitialized = true;
    }

    _createBloomFBOs(w, h) {
        const gl = this.gl;
        const MIP_LEVELS = 5;

        // Intermediate cam_brush output (RGBA32F, LINEAR for bloom sampling)
        this.camBrushTexture = createFloatTexture(gl, w, h);
        gl.bindTexture(gl.TEXTURE_2D, this.camBrushTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        this.camBrushFBO = createFramebuffer(gl, this.camBrushTexture);

        // Bloom copy texture (holds original + bloom after upsample chain)
        this.bloomCopyTexture = createFloatTexture(gl, w, h);
        gl.bindTexture(gl.TEXTURE_2D, this.bloomCopyTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        this.bloomCopyFBO = createFramebuffer(gl, this.bloomCopyTexture);

        // Mip chain (5 levels, each half the previous)
        this.bloomMipTextures = [];
        this.bloomMipFBOs = [];
        let mw = w, mh = h;
        for (let i = 0; i < MIP_LEVELS; i++) {
            mw = Math.max(1, mw >> 1);
            mh = Math.max(1, mh >> 1);
            const tex = createFloatTexture(gl, mw, mh);
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.bindTexture(gl.TEXTURE_2D, null);
            this.bloomMipTextures.push(tex);
            this.bloomMipFBOs.push(createFramebuffer(gl, tex));
        }

        this._bloomWidth = w;
        this._bloomHeight = h;
    }

    _createBloomVAOs() {
        const vertices = new Float32Array([
            -1, -1,  1, -1,  1,  1,
            -1, -1,  1,  1, -1,  1,
        ]);
        this.bloomDownsampleVAO = this._makeQuadVAO(this.bloomDownsampleProgram, vertices);
        this.bloomUpsampleVAO = this._makeQuadVAO(this.bloomUpsampleProgram, vertices);
        this.tonemapVAO = this._makeQuadVAO(this.tonemapProgram, vertices);
    }

    _destroyBloomFBOs() {
        const gl = this.gl;
        if (this.camBrushTexture) { gl.deleteTexture(this.camBrushTexture); this.camBrushTexture = null; }
        if (this.camBrushFBO) { gl.deleteFramebuffer(this.camBrushFBO); this.camBrushFBO = null; }
        if (this.bloomCopyTexture) { gl.deleteTexture(this.bloomCopyTexture); this.bloomCopyTexture = null; }
        if (this.bloomCopyFBO) { gl.deleteFramebuffer(this.bloomCopyFBO); this.bloomCopyFBO = null; }
        for (const tex of this.bloomMipTextures) gl.deleteTexture(tex);
        for (const fbo of this.bloomMipFBOs) gl.deleteFramebuffer(fbo);
        this.bloomMipTextures = [];
        this.bloomMipFBOs = [];
        this._bloomWidth = 0;
        this._bloomHeight = 0;
    }

    _destroyBloomVAOs() {
        const gl = this.gl;
        if (this.bloomDownsampleVAO) { gl.deleteVertexArray(this.bloomDownsampleVAO); this.bloomDownsampleVAO = null; }
        if (this.bloomUpsampleVAO) { gl.deleteVertexArray(this.bloomUpsampleVAO); this.bloomUpsampleVAO = null; }
        if (this.tonemapVAO) { gl.deleteVertexArray(this.tonemapVAO); this.tonemapVAO = null; }
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
        tryset(gl, prog, 'canvas_resolution', [c.canvasWidth, c.canvasHeight]);

        gl.bindVertexArray(this.canvasQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        this.canvasPing = writeIdx;
    }

    renderDisplay(fancyCamera, camera, brightness) {
        if (fancyCamera && camera) {
            if (!this._bloomInitialized) {
                if (!this._bloomInitializing) {
                    this._bloomInitializing = true;
                    this._initBloomResources().then(() => {
                        this._bloomInitializing = false;
                    });
                }
                // Render directly to screen while bloom resources are loading
                this.renderCamBrush(camera, null, brightness);
                return;
            }

            // Recreate bloom FBOs if canvas size changed
            const gl = this.gl;
            if (gl.canvas.width !== this._bloomWidth || gl.canvas.height !== this._bloomHeight) {
                this._destroyBloomFBOs();
                this._destroyBloomVAOs();
                this._createBloomFBOs(gl.canvas.width, gl.canvas.height);
                this._createBloomVAOs();
            }

            // Step 1: Render cam_brush to intermediate RGBA32F FBO
            this.renderCamBrush(camera, this.camBrushFBO, brightness);

            // Step 2: Bloom (downsample + blit + upsample)
            this._bloomPass();

            // Step 3: Tonemap to screen
            this._tonemapPass();
        } else {
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
    }

    renderCamBrush(camera, targetFBO, brightness) {
        const gl = this.gl;
        const prog = this.camBrushProgram;
        const c = this.c;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Bind entity texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.entityTextures[this.entityPing]);
        tryset(gl, prog, 'entity_texture', 0);

        // Set uniforms
        tryset(gl, prog, 'entity_tex_width', c.entityTexWidth);
        tryset(gl, prog, 'entity_count', c.entityCount);
        tryset(gl, prog, 'cohorts', this.config.cohorts);
        tryset(gl, prog, 'sqrt_world_size', c.sqrtWorldSize, 'float');
        tryset(gl, prog, 'cam_pos', [camera.posX, camera.posY]);
        tryset(gl, prog, 'cam_zoom', camera.zoom, 'float');
        tryset(gl, prog, 'window_size', [gl.canvas.width, gl.canvas.height]);
        tryset(gl, prog, 'canvas_resolution', [c.canvasWidth, c.canvasHeight]);
        tryset(gl, prog, 'brightness', brightness, 'float');

        // Additive blending
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        gl.bindVertexArray(this.camBrushVAO);
        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, c.entityCount);

        gl.disable(gl.BLEND);
    }

    // ─── Postprocess passes ──────────────────────────────────────────────────

    _bloomPass() {
        const gl = this.gl;
        const MIP_LEVELS = 5;
        const w = this._bloomWidth;
        const h = this._bloomHeight;

        // --- Downsample chain: camBrushTexture → mip[0] → ... → mip[4] ---
        const dsProg = this.bloomDownsampleProgram;
        gl.useProgram(dsProg);
        gl.bindVertexArray(this.bloomDownsampleVAO);

        for (let i = 0; i < MIP_LEVELS; i++) {
            const srcTex = (i === 0) ? this.camBrushTexture : this.bloomMipTextures[i - 1];
            const sw = (i === 0) ? w : Math.max(1, w >> i);
            const sh = (i === 0) ? h : Math.max(1, h >> i);
            const dstW = Math.max(1, w >> (i + 1));
            const dstH = Math.max(1, h >> (i + 1));

            gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomMipFBOs[i]);
            gl.viewport(0, 0, dstW, dstH);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, srcTex);
            tryset(gl, dsProg, 'source_tex', 0);
            tryset(gl, dsProg, 'source_texel_size', [1.0 / sw, 1.0 / sh]);
            tryset(gl, dsProg, 'is_first_pass', (i === 0) ? 1 : 0);

            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        // --- Blit original HDR into bloomCopyTexture for compositing ---
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.camBrushFBO);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.bloomCopyFBO);
        gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);

        // --- Upsample chain with additive blending: mip[4] → ... → mip[0] → bloomCopyFBO ---
        const usProg = this.bloomUpsampleProgram;
        gl.useProgram(usProg);
        gl.bindVertexArray(this.bloomUpsampleVAO);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);

        for (let i = MIP_LEVELS - 1; i >= 0; i--) {
            const srcTex = this.bloomMipTextures[i];
            const srcW = Math.max(1, w >> (i + 1));
            const srcH = Math.max(1, h >> (i + 1));

            let dstFBO, dstW, dstH;
            if (i === 0) {
                // Final upsample: blend bloom onto original in bloomCopyFBO
                dstFBO = this.bloomCopyFBO;
                dstW = w;
                dstH = h;
            } else {
                dstFBO = this.bloomMipFBOs[i - 1];
                dstW = Math.max(1, w >> i);
                dstH = Math.max(1, h >> i);
            }

            gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
            gl.viewport(0, 0, dstW, dstH);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, srcTex);
            tryset(gl, usProg, 'source_tex', 0);
            tryset(gl, usProg, 'source_texel_size', [1.0 / srcW, 1.0 / srcH]);

            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        gl.disable(gl.BLEND);
    }

    _tonemapPass() {
        const gl = this.gl;
        const prog = this.tonemapProgram;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.bloomCopyTexture);
        tryset(gl, prog, 'source_tex', 0);

        gl.bindVertexArray(this.tonemapVAO);
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
