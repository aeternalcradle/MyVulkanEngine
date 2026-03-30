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

layout(push_constant) uniform PushConstants {
    mat4  model;
    float metallic;
    float roughness;
} push;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    mat3 normalMatrix = transpose(inverse(mat3(push.model)));
    fragWorldPos = worldPos.xyz;
    fragWorldNormal = normalize(normalMatrix * inNormal);
    fragTexCoord = inTexCoord;
}
