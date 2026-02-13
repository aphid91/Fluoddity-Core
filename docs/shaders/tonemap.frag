#version 300 es
precision highp float;
// Tone mapping: brightness scaling + asinh soft compression.
// Applied as final postprocess pass to convert HDR to displayable range.

#define TONEMAP_SOFTNESS 3.0

uniform sampler2D source_tex;
uniform float brightness;

in vec2 uv;
out vec4 fragColor;

void main() {
    fragColor = texture(source_tex, uv);

    float brightness_constant = 30.0;
    // Apply brightness multiplier before tone mapping
    fragColor.xyz *= brightness * brightness_constant;

    // Asinh soft-knee compression: preserves hue, compresses intensity
    float len = length(fragColor.xyz);
    if (len > 0.0) {
        fragColor.xyz *= asinh(len * TONEMAP_SOFTNESS) / (len * TONEMAP_SOFTNESS);
    }
}
