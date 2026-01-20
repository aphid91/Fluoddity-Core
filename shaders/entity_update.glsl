#version 430

layout(local_size_x = 256) in;

struct Entity {
    vec2 pos;
    vec2 vel;
    float size;
    float cohort;
    float padding[2];
    vec4 color;
};  // Total: 48 bytes (12 floats)

layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= entities.length()) return;

    Entity e = entities[idx];
    e.pos += e.vel;
    entities[idx] = e;
}
