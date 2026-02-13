/**
 * Bit-exact JavaScript port of GLSL PCG hash and rule computation
 * from entity_update.frag. Used to reconstruct the rule a specific
 * entity was using, given the config rule and entity index.
 */

// Shared buffer for float↔uint bit reinterpretation
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** Round to float32 precision (matches GLSL float behavior). */
function f32(x) {
    _f32[0] = x;
    return _f32[0];
}

/** Reinterpret float bits as uint32 (matches GLSL floatBitsToUint). */
function floatBitsToUint(f) {
    _f32[0] = f;
    return _u32[0];
}

/** PCG hash — bit-exact match of the GLSL version. */
function pcg_hash(seed) {
    seed = seed >>> 0;
    let state = (Math.imul(seed, 747796405) + 2891336453) >>> 0;
    let word = ((state >>> ((state >>> 28) + 4)) ^ state) >>> 0;
    word = Math.imul(word, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

/** 2D hash: two floats → float in [0, 1]. Matches GLSL hash(vec2). */
function hash(x, y) {
    const ux = floatBitsToUint(x);
    const uy = floatBitsToUint(y);
    const h = pcg_hash((ux ^ pcg_hash(uy)) >>> 0);
    return h / 0xFFFFFFFF;
}

/** 4D hash: two floats → array of 4 floats in [0, 1]. Matches GLSL hash4(vec2). */
function hash4(x, y) {
    return [
        hash(x, y),
        hash(f32(-x + 5.0), f32(-y + 5.0)),
        hash(f32(y - 100.0), f32(x - 100.0)),
        hash(f32(-y + 25.0), f32(-x + 25.0)),
    ];
}

/**
 * Generate 10 random Fourier centers from a seed.
 * Returns an 80-element Float32Array (10 centers × 8 floats each).
 * Layout: [freq.x, freq.y, freq.z, freq.w, amp.x, amp.y, amp.z, amp.w] per center.
 * Matches GLSL generate_random_centers().
 */
export function generateRandomCenters(seed) {
    // seed is already f32-rounded by caller
    const rule = new Float32Array(80);
    for (let i = 0; i < 10; i++) {
        const base = i * 8;
        const h0 = hash(seed, i * 8 + 0);
        const freqScale = f32(1.0 + f32(2.0 * f32(h0 * h0)));
        rule[base + 0] = f32(f32(h0 * 2.0 - 1.0) * freqScale);
        rule[base + 1] = f32(f32(hash(seed, i * 8 + 1) * 2.0 - 1.0) * freqScale);
        rule[base + 2] = f32(f32(hash(seed, i * 8 + 2) * 2.0 - 1.0) * freqScale);
        rule[base + 3] = f32(f32(hash(seed, i * 8 + 3) * 2.0 - 1.0) * freqScale);
        rule[base + 4] = f32(hash(seed, i * 8 + 4) * 2.0 - 1.0);
        rule[base + 5] = f32(hash(seed, i * 8 + 5) * 2.0 - 1.0);
        rule[base + 6] = f32(hash(seed, i * 8 + 6) * 2.0 - 1.0);
        rule[base + 7] = f32(hash(seed, i * 8 + 7) * 2.0 - 1.0);
    }
    return rule;
}

/**
 * Mutate a rule array in-place. Matches GLSL mutate_rule().
 * rule: 80-element array (10 centers × 8 floats)
 * amount: mutation scale
 * cohort: cohort seed value
 */
export function mutateRule(rule, amount, cohort) {
    // Seed from specific center components, matching GLSL:
    // hash(centers[4].frequency.xy + centers[7].amplitude.yx + centers[1].frequency.zw) + cohort
    // centers[4].frequency.xy = rule[32], rule[33]
    // centers[7].amplitude.yx = rule[61], rule[60]
    // centers[1].frequency.zw = rule[10], rule[11]
    // Use f32() to match GLSL float32 addition precision
    const seedX = f32(f32(rule[32] + rule[61]) + rule[10]);
    const seedY = f32(f32(rule[33] + rule[60]) + rule[11]);
    const seed = f32(hash(seedX, seedY) + cohort);

    for (let i = 0; i < 10; i++) {
        const base = i * 8;

        // Amplitude mutation: additive
        // Match GLSL: hash4(-0.5 + vec2(-float(i) + seed, float(i)))
        const ampMut = hash4(f32(-0.5 + f32(-i + seed)), f32(-0.5 + i));
        rule[base + 4] = f32(rule[base + 4] + f32(amount * f32(-1.0 + 2.0 * ampMut[0])));
        rule[base + 5] = f32(rule[base + 5] + f32(amount * f32(-1.0 + 2.0 * ampMut[1])));
        rule[base + 6] = f32(rule[base + 6] + f32(amount * f32(-1.0 + 2.0 * ampMut[2])));
        rule[base + 7] = f32(rule[base + 7] + f32(amount * f32(-1.0 + 2.0 * ampMut[3])));

        // Frequency mutation: multiplicative
        // Match GLSL: 1.0 + amount * 0.5 * (hash(vec2(seed, float(i))) - 0.5)
        const freqMut = f32(1.0 + f32(f32(amount * 0.5) * f32(hash(seed, i) - 0.5)));
        rule[base + 0] = f32(rule[base + 0] * freqMut);
        rule[base + 1] = f32(rule[base + 1] * freqMut);
        rule[base + 2] = f32(rule[base + 2] * freqMut);
        rule[base + 3] = f32(rule[base + 3] * freqMut);
    }
}

/**
 * Compute the exact rule an entity was using, given the config state.
 * Reproduces entity_update.frag lines 186–213.
 *
 * @param {number[]|Float32Array} configRule - 80-element base rule
 * @param {number} ruleSeed - config.rule_seed
 * @param {number} mutationScale - config.mutation_scale
 * @param {number} cohorts - config.cohorts
 * @param {number} entityIndex - index of the entity
 * @param {number} entityCount - total entity count
 * @returns {Float32Array} 80-element mutated rule
 */
export function computeEntityRule(configRule, ruleSeed, mutationScale, cohorts, entityIndex, entityCount) {
    const rule = new Float32Array(80);

    // Copy config rule
    for (let i = 0; i < 80; i++) {
        rule[i] = configRule[i];
    }

    // GLSL: float(config.cohorts) * float(index) / float(entity_count)
    const cohort = f32(f32(f32(cohorts) * f32(entityIndex)) / f32(entityCount));

    // Check if rule is all-zeros (same check as GLSL: centers[0].frequency == vec4(0) && centers[5].amplitude == vec4(0))
    const freq0AllZero = rule[0] === 0 && rule[1] === 0 && rule[2] === 0 && rule[3] === 0;
    const amp5AllZero = rule[44] === 0 && rule[45] === 0 && rule[46] === 0 && rule[47] === 0;

    // cohort seed = rule_seed + floor(cohort), computed in float32 to match GLSL
    const cohortSeed = f32(ruleSeed + Math.floor(cohort));

    if (freq0AllZero && amp5AllZero) {
        const generated = generateRandomCenters(cohortSeed);
        for (let i = 0; i < 80; i++) {
            rule[i] = generated[i];
        }
    }

    mutateRule(rule, mutationScale, cohortSeed);

    return rule;
}
