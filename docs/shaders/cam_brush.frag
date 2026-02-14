#version 300 es
precision highp float;

in vec2 v_uv;
in vec4 v_color;  // (hue, saturation, brightness, alpha)
layout(location = 0) out vec4 fragColor;

uniform float brightness;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float gaussian(vec2 pos, float sigma) {
    float sigma2 = sigma * sigma;
    float norm = 1.0 / (2.0 * 3.14159265359 * sigma2);
    float exponent = -(dot(pos, pos)) / (2.0 * sigma2);
    return norm * exp(exponent);
}

void main() {
    vec2 centered_uv = v_uv - 0.5;

    // Discard outside circular boundary or zero alpha
    if (length(centered_uv) > 0.5 || v_color.w == 0.0) {
        discard;
    }

    float kernel = 0.5 * gaussian(centered_uv, 0.163);

    float BRIGHTNESS_CONSTANT = 3.;
    vec3 rgb = hsv2rgb(v_color.xyz)*brightness*BRIGHTNESS_CONSTANT;
    fragColor = vec4(rgb, v_color.w * kernel);
}
