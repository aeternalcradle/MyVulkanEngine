#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightViewProj;
    vec3 lightDir;
    float ambient;
    vec3 lightColor;
    float lightSize;
    vec3 camPos;
} ubo;

layout(location = 0) out vec3 localPos;

const vec3 cubeVerts[36] = vec3[](
    vec3(-1, -1, -1), vec3( 1,  1, -1), vec3( 1, -1, -1),
    vec3( 1,  1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
    vec3(-1, -1,  1), vec3( 1, -1,  1), vec3( 1,  1,  1),
    vec3( 1,  1,  1), vec3(-1,  1,  1), vec3(-1, -1,  1),
    vec3(-1,  1,  1), vec3(-1,  1, -1), vec3(-1, -1, -1),
    vec3(-1, -1, -1), vec3(-1, -1,  1), vec3(-1,  1,  1),
    vec3( 1,  1,  1), vec3( 1, -1, -1), vec3( 1,  1, -1),
    vec3( 1, -1, -1), vec3( 1,  1,  1), vec3( 1, -1,  1),
    vec3(-1, -1, -1), vec3( 1, -1, -1), vec3( 1, -1,  1),
    vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, -1, -1),
    vec3(-1,  1, -1), vec3( 1,  1,  1), vec3( 1,  1, -1),
    vec3( 1,  1,  1), vec3(-1,  1, -1), vec3(-1,  1,  1)
);

void main() {
    vec3 pos = cubeVerts[gl_VertexIndex];
    localPos = pos;

    mat4 viewNoTranslation = mat4(mat3(ubo.view));
    vec4 clipPos = ubo.proj * viewNoTranslation * vec4(pos, 1.0);
    gl_Position = clipPos.xyww;
}
