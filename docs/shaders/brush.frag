#version 300 es
precision highp float;

in vec2 v_uv;
in vec4 v_pos_vel;
layout(location = 0) out vec4 brush_out;
uniform int frame_count;

float gaussian(vec2 pos, float sigma) {
    float sigma2 = sigma * sigma;
    float norm = 1.0 / (2.0 * 3.14159265359 * sigma2);
    float exponent = -(dot(pos, pos)) / (2.0 * sigma2);
    return norm * exp(exponent);
}

void main() {
    float kernel_func = gaussian(v_uv - 0.5, 0.163);
    if (length(v_uv - 0.5) > 0.5 || frame_count == 0) {
        discard;
    }
    vec2 vel = v_pos_vel.zw;
    brush_out = vec4(vel, 0.01, 1.0) * kernel_func;
}
