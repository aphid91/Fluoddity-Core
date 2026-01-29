#version 430

#define TRAIL_PERSISTENCE 0.9

uniform sampler2D brush_texture;
uniform sampler2D canvas_texture;
uniform int frame_count;

in vec2 uv;
out vec4 canvas_out;

void main() {
    if(frame_count<5){canvas_out=vec4(0,0,0,1);return;}
    vec4 brush_color = texture(brush_texture, uv);
    vec4 canvas_color = texture(canvas_texture, uv);
    canvas_out = canvas_color * TRAIL_PERSISTENCE + (1.0 - TRAIL_PERSISTENCE) * vec4(brush_color.xy,0,1);
}
