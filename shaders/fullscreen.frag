#version 450

layout (location = 0) out vec4 out_color;
layout (set = 0, binding = 0, rgba8) uniform readonly image2D rendered_image;

void main() {
    out_color = imageLoad(rendered_image, ivec2(gl_FragCoord.xy));
}