#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightViewProj;
    vec3 lightDir;
    float ambient;
    vec3 lightColor;
    float lightSize;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec4 fragPosLightSpace;

layout(location = 0) out vec4 outColor;

// 26-sample Poisson disk for high quality sampling
const int NUM_SAMPLES = 25;
const vec2 poissonDisk[25] = vec2[](
    vec2(-0.9784, -0.1868), vec2(-0.9294,  0.2569), vec2(-0.8275, -0.5765),
    vec2(-0.7380,  0.6310), vec2(-0.6094, -0.0522), vec2(-0.5574,  0.3413),
    vec2(-0.4718, -0.7621), vec2(-0.3921,  0.8706), vec2(-0.2650, -0.3694),
    vec2(-0.1720,  0.1361), vec2(-0.0898, -0.8843), vec2( 0.0145,  0.5818),
    vec2( 0.0678, -0.1570), vec2( 0.1506,  0.9170), vec2( 0.2345, -0.5961),
    vec2( 0.3348,  0.2682), vec2( 0.4156, -0.0406), vec2( 0.5037, -0.8364),
    vec2( 0.5695,  0.6033), vec2( 0.6480, -0.4061), vec2( 0.7279,  0.1243),
    vec2( 0.7876, -0.6819), vec2( 0.8589,  0.4297), vec2( 0.9337, -0.1958),
    vec2( 0.9672,  0.7406)
);

const float ORTHO_WIDTH = 16.0;    // 与 C++ ortho(-8, 8, ...) 对应
const float NEAR_PLANE  = 0.1;
const float BIAS        = 0.002;

// Step 1: 搜索周围遮挡体，返回平均遮挡深度。无遮挡返回 -1
float findBlockerDepth(vec2 uv, float zReceiver) {
    float lightSizeUV = ubo.lightSize / ORTHO_WIDTH;
    float searchRadius = lightSizeUV * (zReceiver - NEAR_PLANE) / zReceiver;
    searchRadius = max(searchRadius, lightSizeUV * 0.1);

    float blockerSum = 0.0;
    int   blockerCount = 0;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        vec2  offset = poissonDisk[i] * searchRadius;
        float depth  = texture(shadowMap, uv + offset).r;
        if (depth < zReceiver - BIAS) {
            blockerSum += depth;
            blockerCount++;
        }
    }

    if (blockerCount == 0) return -1.0;
    return blockerSum / float(blockerCount);
}

// Step 2 + 3: 根据 penumbra 半径做 variable-size PCF
float pcfFilter(vec2 uv, float zReceiver, float filterRadius) {
    float shadow = 0.0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        vec2  offset = poissonDisk[i] * filterRadius;
        float depth  = texture(shadowMap, uv + offset).r;
        shadow += (depth >= zReceiver - BIAS) ? 1.0 : 0.0;
    }
    return shadow / float(NUM_SAMPLES);
}

float calcPCSS(vec4 posLS) {
    vec3 proj = posLS.xyz / posLS.w;
    proj.xy = proj.xy * 0.5 + 0.5;

    if (proj.z > 1.0) return 1.0;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) return 1.0;

    float zReceiver = proj.z;
    float lightSizeUV = ubo.lightSize / ORTHO_WIDTH;

    // Step 1: blocker search
    float avgBlockerDepth = findBlockerDepth(proj.xy, zReceiver);
    if (avgBlockerDepth < 0.0) return 1.0;

    // Step 2: penumbra estimation (similar triangles)
    float penumbraRatio = (zReceiver - avgBlockerDepth) / avgBlockerDepth;
    float filterRadius  = lightSizeUV * penumbraRatio;
    filterRadius = clamp(filterRadius, 0.5 / textureSize(shadowMap, 0).x, lightSizeUV * 2.0);

    // Step 3: PCF with variable radius
    return pcfFilter(proj.xy, zReceiver, filterRadius);
}

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);

    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-ubo.lightDir);

    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * ubo.lightColor;

    float shadow = calcPCSS(fragPosLightSpace);

    vec3 finalColor = texColor.rgb * (vec3(ubo.ambient) + shadow * diffuse);

    outColor = vec4(finalColor, texColor.a);
}
