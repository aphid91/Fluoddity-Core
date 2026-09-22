#version 430

// Adds a source texture into whatever framebuffer is bound, using additive
// blending set up on the Python side (blend_func = ONE, ONE).
//
// This is how the rig time-averages the brush without reading it back every
// step. A readback is ~4 MB; over a 2000-step frozen window that would be 8 GB
// of PCIe traffic and would dominate the measurement it is trying to make.
//
// NOTE: this shader is not part of the physics. It never feeds back into the
// simulation, so it cannot perturb the trajectories being measured.

uniform sampler2D src;

in vec2 uv;
out vec4 frag_out;

void main() {
    frag_out = texture(src, uv);
}
