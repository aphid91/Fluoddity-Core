#version 300 es
precision highp float;
// Bloom upsample: 3x3 tent filter.
// Compositing with the destination is handled via GL additive blending,
// so this shader only outputs the upsampled bloom contribution.

#define BLOOM_RADIUS 1.0

uniform sampler2D source_tex;       // Lower-res bloom mip being upsampled
uniform vec2 source_texel_size;     // 1.0 / source (lower-res) resolution

in vec2 uv;
out vec4 fragColor;

void main() {
    // 3x3 tent filter for smooth upsampling
    float r = BLOOM_RADIUS;
    vec3 sum = vec3(0.0);
    sum += texture(source_tex, uv + source_texel_size * vec2(-r, -r)).rgb * 1.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( 0, -r)).rgb * 2.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( r, -r)).rgb * 1.0;
    sum += texture(source_tex, uv + source_texel_size * vec2(-r,  0)).rgb * 2.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( 0,  0)).rgb * 4.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( r,  0)).rgb * 2.0;
    sum += texture(source_tex, uv + source_texel_size * vec2(-r,  r)).rgb * 1.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( 0,  r)).rgb * 2.0;
    sum += texture(source_tex, uv + source_texel_size * vec2( r,  r)).rgb * 1.0;
    sum /= 16.0;

    fragColor = vec4(sum, 1.0);
}
