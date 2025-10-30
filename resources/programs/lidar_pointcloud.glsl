#version 330

// ------------------------ 顶点着色器 ------------------------
#if defined VERTEX_SHADER

layout(location = 0) in vec3 in_position; // 点坐标 (N,3)
layout(location = 1) in vec3 in_color;    // 可选颜色 (N,3)

out vec3 v_color;

void main() {
    // 仅传递到几何着色器
    v_color = in_color;
}

#elif defined GEOMETRY_SHADER

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform float radius = 0.01; // 点渲染半径

in vec3 v_color[];
out vec3 normal;
out vec3 position;
out vec2 fragTexCoord;
out vec3 color;

void main() {
    vec3 p0 = gl_in[0].gl_Position.xyz; // 模型空间坐标
    vec3 col = v_color[0];

    // 法线简单向上
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    vec3 bitangent = cross(N, tangent);

    vec3 quad[4];
    quad[0] = p0 + (-tangent - bitangent) * radius;
    quad[1] = p0 + (tangent - bitangent) * radius;
    quad[2] = p0 + (-tangent + bitangent) * radius;
    quad[3] = p0 + (tangent + bitangent) * radius;

    vec2 texCoords[4] = vec2[4](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(1.0,1.0));

    mat4 mv = m_camera * m_model;
    mat4 mvp = m_proj * mv;

    for(int i=0; i<4; i++){
        normal = mat3(mv) * N;
        position = (mv * vec4(quad[i],1.0)).xyz;
        color = col;
        fragTexCoord = texCoords[i];
        gl_Position = mvp * vec4(quad[i],1.0);
        EmitVertex();
    }
    EndPrimitive();
}

#elif defined FRAGMENT_SHADER

in vec3 normal;
in vec3 position;
in vec2 fragTexCoord;
in vec3 color;

out vec4 out_color;

uniform vec3 lightpos = vec3(1.0,1.0,1.0);
uniform float kA = 0.1;
uniform float kD = 0.8;
uniform float kS = 0.5;
uniform float shininess = 32.0;

void main() {
    // 圆盘裁剪
    if(length(fragTexCoord) > 1.0){
        discard;
    }

    vec3 N = normalize(normal);
    vec3 L = normalize(lightpos - position);
    float lambertian = max(dot(N,L),0.0);

    vec3 V = normalize(-position);
    vec3 R = reflect(-L, N);
    float specular = pow(max(dot(R,V),0.0), shininess);

    vec3 finalColor = color * (kA + kD * lambertian) + vec3(1.0) * kS * specular;
    out_color = vec4(finalColor,1.0);
}

#endif
