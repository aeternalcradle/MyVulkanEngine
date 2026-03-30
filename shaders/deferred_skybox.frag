#version 450

layout(set = 0, binding = 3) uniform sampler2D gPosition;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightViewProj;
    vec3 lightDir;
    float ambient;
    vec3 lightColor;
    float lightSize;
    vec3 camPos;
} ubo;
layout(set = 1, binding = 6) uniform samplerCube envMap;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 posData = texture(gPosition, uv);
    if (posData.a > 0.5) {
        discard;
    }

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, 1.0, 1.0);
    vec3 viewDir = normalize((inverse(ubo.proj) * clip).xyz);

    mat3 rotView = mat3(ubo.view);
    vec3 worldDir = normalize(transpose(rotView) * viewDir);

    vec3 sky = texture(envMap, worldDir).rgb;
    outColor = vec4(sky, 1.0);
}
