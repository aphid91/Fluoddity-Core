/**
 * Auto-detection calibration: measures GPU performance at startup
 * to select optimal world size and physics speed.
 *
 * Uses gl.readPixels() on a 1x1 region to force GPU pipeline flush,
 * which is the reliable way to synchronize in WebGL (gl.finish() is
 * often a no-op on many drivers, especially mobile).
 *
 * This sync is ONLY done during calibration, never during normal rendering.
 */

// Target frame time budget for physics (ms).
// At 60fps total budget is ~16.67ms; we leave headroom for rendering/bloom.
const FRAME_TIME_BUDGET_MS = 15.0;

// Target physics speed (iterations per frame). No use going above this.
const PHYSICS_SPEED_TARGET = 8;

// Available world sizes (must match <option> values in index.html)
const WORLD_SIZES = [0.08, 0.25, 0.5, 1.0];

const WARMUP_FRAMES = 2;
const MEASURE_FRAMES = 5;

// If a single warmup frame exceeds this, skip measuring and move on.
const BAIL_THRESHOLD_MS = 200;

// Scratch buffer for readPixels sync (1 pixel, RGBA float)
const _syncBuf = new Float32Array(4);

/**
 * Force GPU to complete all pending work by reading back a single pixel.
 * This is the only reliable GPU sync method in WebGL — gl.finish() is
 * unreliable across browsers/drivers, especially on mobile.
 */
function gpuSync(gl, system) {
    // Read 1 pixel from the canvas FBO that advance() just wrote to.
    // The FBO is already bound after updateCanvas(), but bind explicitly
    // to be safe in case something changed.
    const fbo = system.canvasFBOs[system.canvasPing];
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, _syncBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

/**
 * Measure the median time (ms) for one physics frame at current settings.
 * Returns null if warmup indicates this setting is way too slow (bail early).
 */
function measureFrameTime(gl, system, physicsIterations) {
    // Warmup with early bail: if a single frame is absurdly slow, don't
    // waste time running the full warmup + measurement cycle
    for (let i = 0; i < WARMUP_FRAMES; i++) {
        const t0 = performance.now();
        for (let j = 0; j < physicsIterations; j++) {
            system.advance();
        }
        gpuSync(gl, system);
        const elapsed = performance.now() - t0;

        if (elapsed > BAIL_THRESHOLD_MS) {
            // This setting is way too slow, bail early
            return elapsed;
        }
    }

    // Measure
    const times = [];
    for (let i = 0; i < MEASURE_FRAMES; i++) {
        const t0 = performance.now();
        for (let j = 0; j < physicsIterations; j++) {
            system.advance();
        }
        gpuSync(gl, system);
        times.push(performance.now() - t0);
    }

    // Return median to reduce outlier influence
    times.sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)];
}

/**
 * Phase 1: Find the largest world size that fits within FRAME_TIME_BUDGET_MS
 * for a single physics iteration. Steps from smallest to largest, breaks on
 * first failure.
 */
function calibrateWorldSize(gl, system, aspectRatio) {
    let bestSize = WORLD_SIZES[0];

    for (const size of WORLD_SIZES) {
        system.reinitGPU(size, aspectRatio);
        const frameTime = measureFrameTime(gl, system, 1);
        console.log(`Calibration: worldSize=${size}, frameTime=${frameTime.toFixed(2)}ms`);

        if (frameTime <= FRAME_TIME_BUDGET_MS) {
            bestSize = size;
        } else {
            break;
        }
    }

    return bestSize;
}

/**
 * Phase 2: Find optimal physics speed at the chosen world size.
 * Uses linear extrapolation from a single-iteration measurement,
 * then verifies with an actual measurement at the estimated speed.
 */
function calibratePhysicsSpeed(gl, system, worldSize, aspectRatio) {
    system.reinitGPU(worldSize, aspectRatio);
    const singleFrameTime = measureFrameTime(gl, system, 1);
    console.log(`Calibration: speed measurement at worldSize=${worldSize}, singleFrame=${singleFrameTime.toFixed(2)}ms`);

    // Linear extrapolation: physics cost scales linearly with iteration count
    let speed = Math.floor(FRAME_TIME_BUDGET_MS / singleFrameTime);
    speed = Math.max(1, Math.min(speed, PHYSICS_SPEED_TARGET));

    // Verify with actual measurement
    if (speed > 1) {
        system.reinitGPU(worldSize, aspectRatio);
        const verifyTime = measureFrameTime(gl, system, speed);
        console.log(`Calibration: verify speed=${speed}, frameTime=${verifyTime.toFixed(2)}ms`);

        if (verifyTime > FRAME_TIME_BUDGET_MS) {
            speed--;
        }
    }

    return speed;
}

/**
 * Run auto-detection calibration.
 * Call after system.init() and setupUI(), before starting the render loop.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {ParticleSystem} system
 * @param {number} aspectRatio
 * @returns {{ worldSize: number, physicsSpeed: number }}
 */
export function calibrate(gl, system, aspectRatio) {
    console.log('Starting performance calibration...');

    const worldSize = calibrateWorldSize(gl, system, aspectRatio);
    console.log(`Calibrated world size: ${worldSize}`);

    const physicsSpeed = calibratePhysicsSpeed(gl, system, worldSize, aspectRatio);
    console.log(`Calibrated physics speed: ${physicsSpeed}`);

    // Final reinit at the chosen world size
    system.reinitGPU(worldSize, aspectRatio);

    console.log('Calibration complete.');
    return { worldSize, physicsSpeed };
}
