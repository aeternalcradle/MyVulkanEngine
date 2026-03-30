#version 450

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    mat4  model;
    float metallic;
    float roughness;
} push;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outAlbedoMetallic;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outPosition;

void main() {
    vec3 albedo = texture(texSampler, fragTexCoord).rgb;

    outAlbedoMetallic = vec4(albedo, push.metallic);
    outNormalRoughness = vec4(normalize(fragWorldNormal), clamp(push.roughness, 0.04, 1.0));
    outPosition = vec4(fragWorldPos, 1.0);
}
