#version 300 es
precision highp float;
precision highp int;

// GPGPU fragment shader: each pixel = one entity (pos.xy, vel.zw)
// Replaces the desktop compute shader (entity_update.glsl)

struct FourierCenter {
    vec4 frequency;
    vec4 amplitude;
};

struct Rule {
    FourierCenter centers[10];
};

uniform Rule config_rule;

struct ConfigData {
    int cohorts;
    float rule_seed;
    float sensor_gain;
    float sensor_angle;
    float sensor_distance;
    float mutation_scale;
    float global_force_mult;
    float drag;
    float strafe_power;
    float axial_force;
    float lateral_force;
    float hazard_rate;
    float trail_persistence;
    float trail_diffusion;
};
uniform ConfigData config;
uniform int initial_conditions;  // 0=Grid, 1=Random, 2=Ring

#define PI 3.1415926

uniform float sqrt_world_size;

uniform sampler2D entity_texture;  // current entity state (ping)
uniform sampler2D canvas_texture;  // trail field
uniform int frame_count;
uniform int entity_count;          // total number of active entities
uniform int entity_tex_width;      // width of entity texture

in vec2 uv;
out vec4 fragColor;

//=========================================================================================
//------------------------------------RANDOM / HASH / NOISE--------------------------------

// PCG hash - bit-exact across all platforms
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash(vec2 co) {
    uvec2 u = uvec2(floatBitsToUint(co.x), floatBitsToUint(co.y));
    uint h = pcg_hash(u.x ^ pcg_hash(u.y));
    return float(h) / float(0xffffffffu);
}

vec4 hash4(vec2 co) {
    return vec4(
        hash(co),
        hash(co * -1.0 + 5.0),
        hash(co.yx - 100.0),
        hash(co.yx * -1.0 + 25.0)
    );
}

vec4 fourier_noise(FourierCenter centers[10], vec4 signals) {
    vec4 result = vec4(0.0);
    for (int i = 0; i < 10; i++) {
        float phase = dot(signals, centers[i].frequency);
        float phase_offset = 2.0 * float(i) * 0.6283 + centers[i].amplitude.w * 3.14159;
        vec4 basis = vec4(
            sin(phase + phase_offset),
            cos(phase + phase_offset * 0.7),
            sin(phase * 2.0 + phase_offset * 1.3),
            cos(phase * 2.0 + phase_offset * 0.5)
        );
        result += centers[i].amplitude * basis;
    }
    return result;
}

FourierCenter[10] generate_random_centers(float seed) {
    FourierCenter centers[10];
    for (int i = 0; i < 10; i++) {
        float freq_scale = 1.0 + 2.0 * pow(hash(vec2(seed, float(i * 8 + 0))), 2.0);
        centers[i].frequency.x = (hash(vec2(seed, float(i * 8 + 0))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.y = (hash(vec2(seed, float(i * 8 + 1))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.z = (hash(vec2(seed, float(i * 8 + 2))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.w = (hash(vec2(seed, float(i * 8 + 3))) * 2.0 - 1.0) * freq_scale;
        centers[i].amplitude.x = hash(vec2(seed, float(i * 8 + 4))) * 2.0 - 1.0;
        centers[i].amplitude.y = hash(vec2(seed, float(i * 8 + 5))) * 2.0 - 1.0;
        centers[i].amplitude.z = hash(vec2(seed, float(i * 8 + 6))) * 2.0 - 1.0;
        centers[i].amplitude.w = hash(vec2(seed, float(i * 8 + 7))) * 2.0 - 1.0;
    }
    return centers;
}

void mutate_rule(inout Rule current_rule, float amount, float cohort) {
    float seed = hash(current_rule.centers[4].frequency.xy + current_rule.centers[7].amplitude.yx + current_rule.centers[1].frequency.zw) + cohort;
    for (int i = 0; i < 10; i++) {
        vec4 amp_mutation = amount * (-1.0 + 2.0 * hash4(-0.5 + vec2(-float(i) + seed, float(i))));
        current_rule.centers[i].amplitude += amp_mutation;
        current_rule.centers[i].frequency *= 1.0 + amount * 0.5 * (hash(vec2(seed, float(i))) - 0.5);
    }
}

//------------------------------------END RANDOM / HASH / NOISE-----------------------------

void pR(inout vec2 p, float a) {
    p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

vec4 get_can(vec2 p) {
    vec2 res = vec2(textureSize(canvas_texture, 0));
    vec2 aspect = vec2(1.0, res.x / res.y);
    vec2 cuv = p / 2.0 * aspect + 0.5;
    return texture(canvas_texture, cuv);
}

vec2 safenorm(vec2 p) {
    return length(p) == 0.0 ? vec2(0.0) : normalize(p);
}

float get_cohort(int index) {
    return float(config.cohorts) * float(index) / float(entity_count);
}

vec4 do_reset(int index) {
    vec2 cansz = vec2(textureSize(canvas_texture,0));
    float cohort_val = get_cohort(index);
    vec2 pos = .019*(vec2(hash(vec2(cohort_val)),hash(vec2(cohort_val+float(index)+2.142)))-.5);
    vec2 vel = 0.01 * 0.005 * (vec2(hash(vec2(cohort_val, float(index))), hash(vec2(cohort_val, pos.y))) * 2.0 - 1.0);
    int cohorts = config.cohorts;
    if(initial_conditions == 0){
        //GRID: position different cohorts at different places in a grid
        float aspect = cansz.y/cansz.x;
        float spots=float(cohorts);
        float spot_rows=ceil(sqrt(spots));
        vec2 gridcell=vec2(int(cohort_val)%int(spot_rows),(int(cohort_val))/int(spot_rows));
        pos+=vec2(1.,aspect)*1.8*((gridcell)/spot_rows+ (1./2.*(1./spot_rows-1.)));
    }
    else if(initial_conditions == 1) {
        //RANDOM: scatter cohorts randomly across the canvas, homogenous start
        pos = vec2(hash(vec2(cohort_val, 1.0)), hash(vec2(cohort_val, 2.0))) * 2.0 - 1.0;
    }
    else{
         //RING: arrange cohorts in a ring pattern
        float angle = cohort_val / float(cohorts) * 2.0 * PI;
        float radius = 0.5*min(cansz.y/cansz.x,cansz.x/cansz.y);
        pos += vec2(cos(angle), sin(angle)) * radius;
    }
    return vec4(pos, vel);
}

vec4 black_box(vec2 L, vec2 R, Rule rule) {
    return fourier_noise(rule.centers, vec4(L, R));
}

vec2 y_reflect(vec2 p) {
    return p * vec2(1.0, -1.0);
}

void calculate_entity_behavior(vec2 L, vec2 R, vec2 axis, Rule rule, out vec2 force, out vec2 strafe) {
    vec2 forward = safenorm(axis);
    vec2 left = vec2(forward.y, -forward.x);

    L = vec2(dot(L, forward), dot(L, left));
    R = vec2(dot(R, forward), dot(R, left));

    vec4 baseterm = black_box(L, R, rule);
    vec4 mirrorterm = black_box(y_reflect(R), y_reflect(L), rule);

    force = baseterm.xy + y_reflect(mirrorterm.xy);
    strafe = baseterm.zw + y_reflect(mirrorterm.zw);

    force = (forward * force.x * config.axial_force) + (left * force.y * config.lateral_force);
    strafe = (forward * strafe.x * config.axial_force) + (left * strafe.y * config.lateral_force);
}

void main() {
    // Map fragment coordinate to linear entity index
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    int index = pixel.y * entity_tex_width + pixel.x;

    // Out-of-bounds entities: output zero
    if (index >= entity_count) {
        fragColor = vec4(0.0);
        return;
    }

    // Read current entity state from texture
    vec4 e = texelFetch(entity_texture, pixel, 0);
    vec2 pos = e.xy;
    vec2 vel = e.zw;

    float cohort = get_cohort(index);
    Rule rule = config_rule;

    // Hazard reset
    bool hazard_reset = config.hazard_rate > hash(vec2(float(index) / float(entity_count), float(frame_count)));

    if (frame_count == 0 || hazard_reset) {
        fragColor = do_reset(index);
        return;
    }

    // Sensor sampling
    float sample_dist = 1.0 / sqrt_world_size * 0.005 * config.sensor_distance;
    vec2 orientation = safenorm(vel);

    vec2 left_sensor_offset = orientation * sample_dist;
    vec2 right_sensor_offset = orientation * sample_dist;
    pR(left_sensor_offset, config.sensor_angle * PI);
    pR(right_sensor_offset, -config.sensor_angle * PI);

    vec4 ltap = get_can(pos + left_sensor_offset);
    vec4 rtap = get_can(pos + right_sensor_offset);

    // Generate random rule if config rule is all zeros
    if (rule.centers[0].frequency == vec4(0) && rule.centers[5].amplitude == vec4(0)) {
        rule = Rule(generate_random_centers(config.rule_seed + floor(cohort)));
    }
    mutate_rule(rule, config.mutation_scale, config.rule_seed + floor(cohort));

    // Rescale sensor values
    float sensor_scaling = sqrt_world_size * 38.855 * config.sensor_gain;
    ltap *= sensor_scaling;
    rtap *= sensor_scaling;

    // Compute entity action
    vec2 strafe = vec2(0.0);
    vec2 force = vec2(0.0);
    calculate_entity_behavior(ltap.xy, rtap.xy, orientation, rule, force, strafe);

    // Rescale output forces
    force *= 1.0 / sqrt_world_size * config.global_force_mult / 400.0;
    strafe *= 1.0 / sqrt_world_size * config.global_force_mult / 20.0;

    // Accelerate
    vel = vel * config.drag + force;

    // Move
    pos += vel;
    pos += strafe * config.strafe_power;

    // Wrap from -1 to 1
    pos = 2.0 * (fract(pos / 2.0 - 0.5) - 0.5);

    fragColor = vec4(pos, vel);
}
