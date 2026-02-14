#version 300 es
precision highp float;
// Bloom downsample: 4-tap bilinear box filter with brightness threshold.
// First pass applies threshold to extract bright regions.
// Subsequent passes just downsample.

#define BLOOM_THRESHOLD 0.18

uniform sampler2D source_tex;
uniform vec2 source_texel_size;  // 1.0 / source resolution
uniform int is_first_pass;

in vec2 uv;
out vec4 fragColor;

void main() {
    // 4-tap bilinear downsample (sample between texels for free filtering)
    vec3 a = texture(source_tex, uv + source_texel_size * vec2(-0.5, -0.5)).rgb;
    vec3 b = texture(source_tex, uv + source_texel_size * vec2( 0.5, -0.5)).rgb;
    vec3 c = texture(source_tex, uv + source_texel_size * vec2(-0.5,  0.5)).rgb;
    vec3 d = texture(source_tex, uv + source_texel_size * vec2( 0.5,  0.5)).rgb;

    vec3 color = (a + b + c + d) * 0.25;

    if (is_first_pass != 0) {
        float brightness = max(color.r, max(color.g, color.b));
        color *= max(0.0, brightness - BLOOM_THRESHOLD) / max(brightness, 0.0001);
    }

    fragColor = vec4(color, 1.0);
}
