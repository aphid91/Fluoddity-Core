#version 300 es
precision highp float;

uniform vec2 canvas_resolution;
uniform sampler2D entity_texture;
uniform int entity_tex_width;

// Per-vertex attributes (quad corners)
in vec2 a_offset;   // quad offset: e.g. (-1,-1), (1,-1), (1,1), (-1,1)
in vec2 a_uv;       // quad UV: (0,0), (1,0), (1,1), (0,1)

uniform float sqrt_world_size;

out vec2 v_uv;
out vec4 v_pos_vel;

void main() {
    // Map gl_InstanceID to texel coordinate in entity texture
    ivec2 tc = ivec2(gl_InstanceID % entity_tex_width, gl_InstanceID / entity_tex_width);
    vec4 entity = texelFetch(entity_texture, tc, 0);

    vec2 entity_pos = entity.xy;
    vec2 entity_vel = entity.zw;
    float size = 0.0015 / sqrt_world_size;

    vec2 vertex_pos = entity_pos + a_offset * size;

    gl_Position = vec4(vertex_pos, 0.0, 1.0) * vec4(1.0, canvas_resolution.x / canvas_resolution.y, 1.0, 1.0);

    v_uv = a_uv;
    v_pos_vel = vec4(entity_pos, entity_vel);
}
