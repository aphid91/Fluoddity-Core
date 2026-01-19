#version 430

in vec2 texcoord;

uniform sampler2D brush_tex;
uniform sampler2D can_tex;
out vec4 can_out;

// Draw trail mode uniforms
uniform bool draw_mode;
uniform vec2 mouse;
uniform vec2 previous_mouse;
uniform float draw_size;
uniform float draw_power;

// Boundary conditions
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP

// Tiling mode
uniform bool tiling_mode;

// SYNCHRONIZED: This struct must match entity_update.glsl
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag
struct PhysicsSetting {
    float slider_value;
    float min_value;
    float max_value;
    float x_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float y_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float cohort_sweep; // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
};

uniform PhysicsSetting TRAIL_PERSISTENCE_SETTING;
uniform PhysicsSetting TRAIL_DIFFUSION_SETTING;

#define COHORTS 64

// SYNCHRONIZED: This function must match entity_update.glsl and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py
float calculate_setting(PhysicsSetting setting, vec2 pos, float cohort){
    //if no sweep modes are active, just return slider value
    if(setting.y_sweep == 0.0 && setting.cohort_sweep == 0.0 && setting.x_sweep == 0.0)
        {return setting.slider_value;}
    //otherwise calculate parameter sweeps
    pos = (pos+1)/2.;//convert to 0..1 for use as a mix coefficient
    cohort = cohort / COHORTS; //convert to 0..1 for mixing

    // Count active sweeps and accumulate results
    float result = 0;
    int active_sweeps = 0;
    if(setting.x_sweep != 0.0) {
        // For inverse sweep (x_sweep < 0), swap min and max
        if(setting.x_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.x);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.x);
        }
        active_sweeps++;
    }
    if(setting.y_sweep != 0.0) {
        // For inverse sweep (y_sweep < 0), swap min and max
        if(setting.y_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.y);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.y);
        }
        active_sweeps++;
    }
    if(setting.cohort_sweep != 0.0) {
        // For inverse sweep (cohort_sweep < 0), swap min and max
        if(setting.cohort_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, cohort);
        } else {
            result += mix(setting.max_value, setting.min_value, cohort);
        }
        active_sweeps++;
    }

    // Average the results to keep within min/max range
    return active_sweeps > 0 ? result / float(active_sweeps) : setting.slider_value;
}

vec4 getCan(vec2 p, sampler2D sam) {
    vec2 uv = (BOUNDARY_CONDITIONS_MODE == 2) ? fract(p) : p;
    return texture(sam, uv);
}

vec4 getBlur(vec2 pos, sampler2D sam,float diffusion_constant) {
    ivec2 imsz = textureSize(sam, 0);
    vec3 off = vec3(1. / vec2(imsz), 0);
    vec2 np = pos + off.zy;
    vec2 sp = pos - off.zy;
    vec2 wp = pos - off.xz;
    vec2 ep = pos + off.xz;
    vec4 nc = getCan(np, sam);
    vec4 sc = getCan(sp, sam);
    vec4 wc = getCan(wp, sam);
    vec4 ec = getCan(ep, sam);
    float K = diffusion_constant;
    return (getCan(pos, sam) * K + nc + sc + wc + ec) / (4. + K);
}

// Gaussian kernel for draw trail mode
float draw_kernel(float distance, float size) {
    // Gaussian: exp(-distance^2 / (2 * sigma^2))
    // Using size as sigma
    float sigma = size;
    return exp(-distance * distance / (2.0 * sigma * sigma));
}

void main() {
    vec4 brush_color = texture(brush_tex, texcoord);
    vec4 can_color;
    float TRAIL_DIFFUSION = calculate_setting(TRAIL_DIFFUSION_SETTING,texcoord*2.-1,0);
    if(TRAIL_DIFFUSION>0){
        TRAIL_DIFFUSION= TRAIL_DIFFUSION*TRAIL_DIFFUSION;//better scaling for slider
        TRAIL_DIFFUSION = 4/(pow(5,(TRAIL_DIFFUSION))-1);//better scaling for slider
        can_color = getBlur(texcoord, can_tex,TRAIL_DIFFUSION);
    }
    else{
        can_color = texture(can_tex,texcoord);
    }

    // Convert texcoord from [0,1] to [-1,1] for position-based sweeps
    vec2 world_pos = texcoord * 2.0 - 1.0;
    float trail_persistence = calculate_setting(TRAIL_PERSISTENCE_SETTING, world_pos, 0.0);

    can_out = can_color * trail_persistence + (1 - trail_persistence) * brush_color;

    // Draw trail mode: add velocity based on mouse drag
    if (draw_mode && draw_power > 0.0) {
        float distance_to_mouse;

        // Calculate velocity to add based on mouse movement
        vec2 mouse_velocity = (mouse - previous_mouse);

        if (tiling_mode) {
            // In tiling mode, check 9-cell neighborhood (3x3) for wrapped distance
            // This allows trail drawing across wrapped edges/corners
            float min_distance = 999.0;
            vec2 min_velocity = vec2(999);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    vec2 wrapped_mouse = mouse + vec2(dx, dy);
                    float dist = length(texcoord - wrapped_mouse);
                    min_distance = min(min_distance, dist);
                    vec2 vel = (wrapped_mouse-previous_mouse);
                    min_velocity = length(vel)<length(min_velocity)?vel:min_velocity;
                }
            }
            distance_to_mouse = min_distance;
            mouse_velocity = min_velocity;
        } else {
            // Normal mode: direct distance calculation
            distance_to_mouse = length(texcoord - mouse);
        }

        mouse_velocity *= draw_power/5;

        // Apply Gaussian kernel and add to velocity channels (RG)
        float kernel_weight = draw_kernel(distance_to_mouse, draw_size);
        can_out.xy += mouse_velocity * kernel_weight/draw_size*(1-trail_persistence);
    }
}
