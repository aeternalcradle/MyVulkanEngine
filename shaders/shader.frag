#version 450

// ---- Uniforms ----
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

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 0, binding = 2) uniform samplerCube irradianceMap;
layout(set = 0, binding = 3) uniform samplerCube prefilterMap;
layout(set = 0, binding = 4) uniform sampler2D   brdfLUT;
layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    mat4  model;
    float metallic;
    float roughness;
} push;

// ---- Inputs ----
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec4 fragPosLightSpace;
layout(location = 4) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// ================================================================
//  PCSS (unchanged from original)
// ================================================================

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

const float ORTHO_WIDTH = 16.0;
const float NEAR_PLANE  = 0.1;
const float BIAS        = 0.002;

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

    float avgBlockerDepth = findBlockerDepth(proj.xy, zReceiver);
    if (avgBlockerDepth < 0.0) return 1.0;

    float penumbraRatio = (zReceiver - avgBlockerDepth) / avgBlockerDepth;
    float filterRadius  = lightSizeUV * penumbraRatio;
    filterRadius = clamp(filterRadius, 0.5 / textureSize(shadowMap, 0).x, lightSizeUV * 2.0);

    return pcfFilter(proj.xy, zReceiver, filterRadius);
}

// ================================================================
//  PBR: Cook-Torrance BRDF
// ================================================================

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ================================================================
//  Tone Mapping: ACES Filmic (Stephen Hill fit — outputs linear)
// ================================================================

const mat3 ACESInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

const mat3 ACESOutputMat = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 ACESFitted(vec3 color) {
    color = ACESInputMat * color;
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;
    return clamp(color, 0.0, 1.0);
}

// ================================================================
//  Main
// ================================================================

void main() {
    vec3 albedo = texture(texSampler, fragTexCoord).rgb;
    float metallic  = push.metallic;
    float roughness = push.roughness;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.camPos - fragWorldPos);
    vec3 L = normalize(-ubo.lightDir);
    vec3 H = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3  numerator   = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3  specular    = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);

    float shadow = calcPCSS(fragPosLightSpace);

    // Direct lighting: diffuse + specular, modulated by shadow
    vec3 Lo = (kD * albedo / PI + specular) * ubo.lightColor * NdotL * shadow;

    // IBL ambient lighting
    float NdotV = max(dot(N, V), 0.0);
    vec3 kS_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD_ibl = (vec3(1.0) - kS_ibl) * (1.0 - metallic);

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuseIBL = kD_ibl * irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefilteredColor * (kS_ibl * brdf.x + brdf.y);

    vec3 ambientColor = diffuseIBL + specularIBL;

    vec3 color = ambientColor + Lo;

    // ACES Filmic Tone Mapping (outputs linear; SRGB swapchain handles gamma)
    color = ACESFitted(color);

    outColor = vec4(color, 1.0);
}
