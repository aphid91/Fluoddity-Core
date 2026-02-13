#version 300 es
precision highp float;

#define SPRITE_SIZE 1.5

uniform sampler2D entity_texture;
uniform int entity_tex_width;
uniform int entity_count;
uniform int cohorts;
uniform float sqrt_world_size;

// Camera uniforms
uniform vec2 cam_pos;
uniform float cam_zoom;
uniform vec2 window_size;
uniform vec2 canvas_resolution;

out vec2 v_uv;
out vec4 v_color;

// PCG hash (matches entity_update.frag)
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash_float(uint x) {
    return float(pcg_hash(x)) / float(0xffffffffu);
}

void main() {
    int instance_id = gl_InstanceID;
    int vertex_id = gl_VertexID;

    // Read entity from texture (same pattern as brush.vert)
    ivec2 tc = ivec2(instance_id % entity_tex_width, instance_id / entity_tex_width);
    vec4 entity = texelFetch(entity_texture, tc, 0);
    vec2 entity_pos = entity.xy;

    // Particle size
    float size = SPRITE_SIZE * 0.0015 / sqrt_world_size;

    // Cohort color: hash the cohort index for a deterministic hue
    int cohort_index = int(floor(float(instance_id) * float(cohorts) / float(entity_count)));
    float hue = hash_float(uint(cohort_index) * 2654435761u);
    v_color = vec4(hue, 0.8, 1.0, 0.045);

    // Quad geometry indexed by gl_VertexID (TRIANGLE_FAN order)
    vec2 offsets[4] = vec2[4](
        vec2(-size, -size),
        vec2( size, -size),
        vec2( size,  size),
        vec2(-size,  size)
    );
    vec2 uv_coords[4] = vec2[4](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0)
    );

    // Aspect ratio correction (mirrors reference cam_brush.vert)
    float tex_aspect = canvas_resolution.x / canvas_resolution.y;
    float window_aspect = window_size.x / window_size.y;

    vec2 scale;
    if (tex_aspect > window_aspect) {
        scale = vec2(1.0, window_aspect / tex_aspect);
    } else {
        scale = vec2(tex_aspect / window_aspect, 1.0);
    }
    scale /= cam_zoom;

    // Frustum culling on particle center
    vec2 center_ndc = entity_pos * vec2(1.0, tex_aspect) * scale;
    center_ndc -= cam_pos * vec2(1.0, -1.0) / cam_zoom;
    float max_extent = size * tex_aspect * max(scale.x, scale.y);

    bool visible = (center_ndc.x + max_extent >= -1.0 && center_ndc.x - max_extent <= 1.0 &&
                    center_ndc.y + max_extent >= -1.0 && center_ndc.y - max_extent <= 1.0);

    // Vertex position
    vec2 vertex_pos = entity_pos + (visible ? offsets[vertex_id] : vec2(0.0));

    // Apply camera transform
    vec2 ndc = vertex_pos * vec2(1.0, tex_aspect) * scale;
    ndc -= cam_pos * vec2(1.0, -1.0) / cam_zoom;

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = uv_coords[vertex_id];
}
