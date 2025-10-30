#version 330

in vec3 frag_color;
out vec4 color;

void main() {
    color = vec4(frag_color, 1.0); // 输出颜色，alpha=1
}
