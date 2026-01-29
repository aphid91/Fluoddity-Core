#version 430

#define TRAIL_PERSISTENCE 0.9819999933242798
#define TRAIL_DIFFUSION 1.
uniform sampler2D brush_texture;
uniform sampler2D canvas_texture;
uniform int frame_count;

in vec2 uv;
out vec4 canvas_out;
vec4 getCan(vec2 p, sampler2D sam) {
    vec2 uv = fract(p);
    return texture(sam, uv);
}

vec4 getBlur(vec2 pos, sampler2D sam,float diffusion_constant) {
    ivec2 imsz = textureSize(sam, 0);
    vec3 off = vec3(1. / vec2(imsz), 0);
    vec2 np = pos + off.zy;
    vec2 sp = pos - off.zy;
    vec2 wp = pos - off.xz;
    vec2 ep = pos + off.xz;
    vec4 nc = getCan(np, sam);
    vec4 sc = getCan(sp, sam);
    vec4 wc = getCan(wp, sam);
    vec4 ec = getCan(ep, sam);
    float K = diffusion_constant;
    return (getCan(pos, sam) * K + nc + sc + wc + ec) / (4. + K);
}
void main() {
    if(frame_count<2){canvas_out=vec4(0,0,0,1);return;}
    vec4 brush_color = texture(brush_texture, uv);
    vec4 canvas_color = getBlur(uv,canvas_texture,TRAIL_DIFFUSION);//texture(canvas_texture, uv);
    canvas_out = canvas_color * TRAIL_PERSISTENCE + (1.0 - TRAIL_PERSISTENCE) * vec4(brush_color.xy,0,1);
}
