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

layout(set = 0, binding = 1) uniform sampler2D gAlbedoMetallic;
layout(set = 0, binding = 2) uniform sampler2D gNormalRoughness;
layout(set = 0, binding = 3) uniform sampler2D gPosition;
layout(set = 1, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 2) uniform samplerCube irradianceMap;
layout(set = 1, binding = 3) uniform samplerCube prefilterMap;
layout(set = 1, binding = 4) uniform sampler2D brdfLUT;
layout(set = 1, binding = 5) uniform sampler2D ssaoMap;
layout(set = 1, binding = 6) uniform samplerCube envMap;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

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
const int SSR_MAX_STEPS = 48;
const float SSR_STEP_SIZE = 0.25;
const float SSR_MAX_DISTANCE = 24.0;
const float SSR_THICKNESS = 0.25;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 1e-6);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(denom, 1e-6);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float findBlockerDepth(vec2 uv, float zReceiver) {
    float lightSizeUV = ubo.lightSize / ORTHO_WIDTH;
    float searchRadius = lightSizeUV * (zReceiver - NEAR_PLANE) / zReceiver;
    searchRadius = max(searchRadius, lightSizeUV * 0.1);

    float blockerSum = 0.0;
    int blockerCount = 0;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        vec2 offset = poissonDisk[i] * searchRadius;
        float depth = texture(shadowMap, uv + offset).r;
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
        vec2 offset = poissonDisk[i] * filterRadius;
        float depth = texture(shadowMap, uv + offset).r;
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

bool traceSSR(vec3 worldPos, vec3 worldNormal, vec3 viewDir,
              out vec2 hitUV, out float hitConfidence)
{
    vec3 reflDir = normalize(reflect(-viewDir, worldNormal));

    // Reflection heading away from camera frustum is usually unstable in SSR.
    if (reflDir.z > 0.995) {
        hitUV = vec2(0.0);
        hitConfidence = 0.0;
        return false;
    }

    float travel = SSR_STEP_SIZE;
    for (int i = 0; i < SSR_MAX_STEPS; ++i) {
        if (travel > SSR_MAX_DISTANCE) break;

        vec3 sampleWorldPos = worldPos + reflDir * travel;
        vec4 clip = ubo.proj * ubo.view * vec4(sampleWorldPos, 1.0);
        if (clip.w <= 1e-5) {
            travel += SSR_STEP_SIZE;
            continue;
        }

        vec3 ndc = clip.xyz / clip.w;
        vec2 sampleUV = ndc.xy * 0.5 + 0.5;

        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0)
            break;

        vec4 gPos = texture(gPosition, sampleUV);
        if (gPos.a > 0.5) {
            float sceneViewZ  = (ubo.view * vec4(gPos.xyz, 1.0)).z;
            float sampleViewZ = (ubo.view * vec4(sampleWorldPos, 1.0)).z;
            float dz = abs(sceneViewZ - sampleViewZ);

            if (dz < SSR_THICKNESS) {
                float edgeFade = 1.0 - clamp(max(abs(sampleUV.x - 0.5), abs(sampleUV.y - 0.5)) * 2.0, 0.0, 1.0);
                float stepFade = 1.0 - float(i) / float(SSR_MAX_STEPS);
                hitUV = sampleUV;
                hitConfidence = edgeFade * stepFade;
                return true;
            }
        }

        travel += SSR_STEP_SIZE;
    }

    hitUV = vec2(0.0);
    hitConfidence = 0.0;
    return false;
}

void main() {
    vec4 albedoMetallic = texture(gAlbedoMetallic, uv);
    vec4 normalRoughness = texture(gNormalRoughness, uv);
    vec3 worldPos = texture(gPosition, uv).xyz;

    vec3 albedo = albedoMetallic.rgb;
    float metallic = clamp(albedoMetallic.a, 0.0, 1.0);
    vec3 N = normalize(normalRoughness.xyz);
    float roughness = clamp(normalRoughness.a, 0.04, 1.0);

    vec3 V = normalize(ubo.camPos - worldPos);
    vec3 L = normalize(-ubo.lightDir);
    vec3 H = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);
    vec4 fragPosLightSpace = ubo.lightViewProj * vec4(worldPos, 1.0);
    float shadow = calcPCSS(fragPosLightSpace);

    vec3 Lo = (kD * albedo / PI + specular) * ubo.lightColor * NdotL * shadow;

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

    vec2 hitUV;
    float hitConfidence;
    bool hit = traceSSR(worldPos, N, V, hitUV, hitConfidence);
   

    float ao = texture(ssaoMap, uv).r;
    vec3 ambient = (diffuseIBL + specularIBL) * ao;
    vec3 color = ambient + Lo;

    color = ACESFitted(color);

    outColor = vec4(color, 1.0);
}
