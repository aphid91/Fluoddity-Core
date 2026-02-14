#version 300 es
precision highp float;
// Tone mapping: brightness scaling + asinh soft compression.
// Applied as final postprocess pass to convert HDR to displayable range.

#define TONEMAP_SOFTNESS 3.9

uniform sampler2D source_tex;

in vec2 uv;
out vec4 fragColor;
#define BRIGHTNESS_CONSTANT 2.
void main() {
    fragColor = texture(source_tex, uv);

    // Asinh soft-knee compression: preserves hue, compresses intensity
    float len = length(fragColor.xyz);
    if (len > 0.0) {
        fragColor.xyz *= BRIGHTNESS_CONSTANT*asinh(len * TONEMAP_SOFTNESS) / (len * TONEMAP_SOFTNESS);
    }
}
