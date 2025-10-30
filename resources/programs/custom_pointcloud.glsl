#version 330

// 顶点输入
in vec3 in_position; // 点坐标
in vec3 in_color;    // 点颜色

// Uniforms
uniform mat4 m_model;   // 点云局部到世界变换
uniform mat4 m_camera;  // 摄像机视图矩阵
uniform mat4 m_proj;    // 投影矩阵
uniform float point_size = 2.0; // 点大小

// 输出给片段着色器
out vec3 frag_color;

void main() {
    frag_color = in_color;
    gl_Position = m_proj * m_camera * m_model * vec4(in_position, 1.0);
    gl_PointSize = point_size;
}
