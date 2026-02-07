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

const WORLD_SIZE = 0.25 ;
const SQRT_WORLD_SIZE = Math.sqrt(WORLD_SIZE);
const ENTITY_COUNT = Math.floor(600000 * WORLD_SIZE);
const CANVAS_DIM = Math.floor(1024 * SQRT_WORLD_SIZE);

// Entity texture dimensions: smallest square that fits ENTITY_COUNT pixels
const ENTITY_TEX_WIDTH = Math.ceil(Math.sqrt(ENTITY_COUNT));
const ENTITY_TEX_HEIGHT = Math.ceil(ENTITY_COUNT / ENTITY_TEX_WIDTH);

export { ENTITY_COUNT, CANVAS_DIM, ENTITY_TEX_WIDTH, ENTITY_TEX_HEIGHT, SQRT_WORLD_SIZE };

export class ParticleSystem {
    constructor(gl, config, aspectRatio) {
        this.gl = gl;
        this.config = config;
        this.frameCount = 0;

        // Canvas dimensions adjusted for viewport aspect ratio
        // sqrt(aspect) factor preserves total pixel area
        const sqrtAspect = Math.sqrt(aspectRatio);
        this.canvasWidth = Math.floor(CANVAS_DIM * sqrtAspect);
        this.canvasHeight = Math.floor(CANVAS_DIM / sqrtAspect);

        // Programs (set in init())
        this.entityUpdateProgram = null;
        this.brushProgram = null;
        this.canvasUpdateProgram = null;
        this.cameraProgram = null;

        // Textures and FBOs (set in init())
        this.entityTextures = [null, null];   // ping-pong
        this.entityFBOs = [null, null];
        this.entityPing = 0;

        this.brushTexture = null;
        this.brushFBO = null;

        this.canvasTextures = [null, null];    // ping-pong
        this.canvasFBOs = [null, null];
        this.canvasPing = 0;

        // VAOs
        this.fullscreenQuadVAO = null;
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

        // Create textures
        this._createTextures();

        // Create VAOs
        this._createFullscreenQuadVAO();
        this._createBrushVAO();
        this._createCameraVAO();
    }

    _createTextures() {
        const gl = this.gl;

        // Entity textures (ping-pong): ENTITY_TEX_WIDTH × ENTITY_TEX_HEIGHT, RGBA32F, NEAREST
        for (let i = 0; i < 2; i++) {
            this.entityTextures[i] = createFloatTexture(gl, ENTITY_TEX_WIDTH, ENTITY_TEX_HEIGHT);
            this.entityFBOs[i] = createFramebuffer(gl, this.entityTextures[i]);
        }

        // Brush texture
        this.brushTexture = createFloatTexture(gl, this.canvasWidth, this.canvasHeight);
        this.brushFBO = createFramebuffer(gl, this.brushTexture);

        // Canvas textures (ping-pong)
        for (let i = 0; i < 2; i++) {
            this.canvasTextures[i] = createFloatTexture(gl, this.canvasWidth, this.canvasHeight);
            this.canvasFBOs[i] = createFramebuffer(gl, this.canvasTextures[i]);
        }
    }

    _createFullscreenQuadVAO() {
        const gl = this.gl;

        // Fullscreen quad: 2 triangles
        const vertices = new Float32Array([
            -1, -1,
             1, -1,
             1,  1,
            -1, -1,
             1,  1,
            -1,  1,
        ]);

        // Create VAO for entity update program
        this.fullscreenQuadVAO = this._makeQuadVAO(this.entityUpdateProgram, vertices);

        // Create VAO for canvas update program (same geometry, different program)
        this.canvasQuadVAO = this._makeQuadVAO(this.canvasUpdateProgram, vertices);

        // Create VAO for camera program
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

        // 4 vertices for a TRIANGLE_FAN quad
        // Each vertex: offset (vec2) + uv (vec2)
        const quadData = new Float32Array([
            // offset.x, offset.y, uv.x, uv.y
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

    _createCameraVAO() {
        // Already created in _createFullscreenQuadVAO
    }

    advance() {
        this.updateEntities();
        this.createBrush();
        this.updateCanvas();
        this.frameCount++;
    }

    updateEntities() {
        const gl = this.gl;
        const prog = this.entityUpdateProgram;
        const readIdx = this.entityPing;
        const writeIdx = 1 - readIdx;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.entityFBOs[writeIdx]);
        gl.viewport(0, 0, ENTITY_TEX_WIDTH, ENTITY_TEX_HEIGHT);

        // Bind entity texture (ping) to unit 0
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.entityTextures[readIdx]);
        tryset(gl, prog, 'entity_texture', 0);

        // Bind canvas texture to unit 1
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.canvasTextures[this.canvasPing]);
        tryset(gl, prog, 'canvas_texture', 1);

        // Set uniforms
        setConfigUniforms(gl, prog, this.config);
        setRuleUniforms(gl, prog, this.config.rule);
        tryset(gl, prog, 'frame_count', this.frameCount);
        tryset(gl, prog, 'entity_count', ENTITY_COUNT);
        tryset(gl, prog, 'entity_tex_width', ENTITY_TEX_WIDTH);

        // Draw fullscreen quad
        gl.bindVertexArray(this.fullscreenQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Swap ping-pong
        this.entityPing = writeIdx;
    }

    createBrush() {
        const gl = this.gl;
        const prog = this.brushProgram;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.brushFBO);
        gl.viewport(0, 0, this.canvasWidth, this.canvasHeight);

        // Clear brush texture
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Additive blending
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        // Bind entity texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.entityTextures[this.entityPing]);
        tryset(gl, prog, 'entity_texture', 0);

        // Set uniforms
        tryset(gl, prog, 'canvas_resolution', [this.canvasWidth, this.canvasHeight]);
        tryset(gl, prog, 'frame_count', this.frameCount);
        tryset(gl, prog, 'entity_tex_width', ENTITY_TEX_WIDTH);

        // Instanced draw: 4 vertices (TRIANGLE_FAN) × ENTITY_COUNT instances
        gl.bindVertexArray(this.brushVAO);
        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, ENTITY_COUNT);

        gl.disable(gl.BLEND);
    }

    updateCanvas() {
        const gl = this.gl;
        const prog = this.canvasUpdateProgram;
        const readIdx = this.canvasPing;
        const writeIdx = 1 - readIdx;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.canvasFBOs[writeIdx]);
        gl.viewport(0, 0, this.canvasWidth, this.canvasHeight);

        // Bind brush texture to unit 0
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
        tryset(gl, prog, 'brush_texture', 0);

        // Bind canvas texture (ping) to unit 1
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.canvasTextures[readIdx]);
        tryset(gl, prog, 'canvas_texture', 1);

        // Set uniforms
        setConfigUniforms(gl, prog, this.config);
        tryset(gl, prog, 'frame_count', this.frameCount);

        // Draw fullscreen quad
        gl.bindVertexArray(this.canvasQuadVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Swap ping-pong
        this.canvasPing = writeIdx;
    }

    renderDisplay() {
        const gl = this.gl;
        const prog = this.cameraProgram;

        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);  // render to screen
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Bind brush texture
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
}
