#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;
class Pipeline;
struct RenderObject;

class SSAO {
public:
    static constexpr int   KERNEL_SIZE     = 64;
    static constexpr int   NOISE_DIM       = 4;
    static constexpr float DEFAULT_RADIUS  = 0.5f;
    static constexpr float DEFAULT_BIAS    = 0.025f;

    VkImageView blurredAOView = VK_NULL_HANDLE;
    VkSampler   aoSampler     = VK_NULL_HANDLE;

    void create(VulkanContext& ctx, Pipeline& pipeline,
                uint32_t width, uint32_t height);
    void destroy(VulkanContext& ctx);
    void resize(VulkanContext& ctx, uint32_t width, uint32_t height);

    void updateParams(const glm::mat4& proj, const glm::vec2& screenSize);
    void recordPrePass(VkCommandBuffer cmd, VkDescriptorSet frameDescSet,
                       const std::vector<RenderObject>& objects);
    void recordCompute(VkCommandBuffer cmd);

private:
    uint32_t width_  = 0;
    uint32_t height_ = 0;

    // Pre-pass render pass & pipeline
    VkRenderPass     prePassRenderPass     = VK_NULL_HANDLE;
    VkPipelineLayout prePassPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       prePassPipeline       = VK_NULL_HANDLE;
    VkFramebuffer    prePassFramebuffer    = VK_NULL_HANDLE;

    // Depth (pre-pass output, sampled by compute)
    VkImage        depthImage   = VK_NULL_HANDLE;
    VmaAllocation depthMem     = VK_NULL_HANDLE;
    VkImageView    depthView    = VK_NULL_HANDLE;
    VkFormat       depthFormat_ = VK_FORMAT_UNDEFINED;

    // View-space normals (pre-pass output, sampled by compute)
    VkImage        normalImage  = VK_NULL_HANDLE;
    VmaAllocation normalMem    = VK_NULL_HANDLE;
    VkImageView    normalView   = VK_NULL_HANDLE;

    // Raw AO (SSAO compute output, blur compute input)
    VkImage        rawAOImage   = VK_NULL_HANDLE;
    VmaAllocation rawAOMem     = VK_NULL_HANDLE;
    VkImageView    rawAOView    = VK_NULL_HANDLE;

    // Blurred AO (blur compute output, main pass input)
    VkImage        blurredAOImage = VK_NULL_HANDLE;
    VmaAllocation blurredAOMem   = VK_NULL_HANDLE;

    // 4x4 noise texture (random tangent-space rotation vectors)
    VkImage        noiseImage   = VK_NULL_HANDLE;
    VmaAllocation noiseMem     = VK_NULL_HANDLE;
    VkImageView    noiseView    = VK_NULL_HANDLE;

    // Kernel UBO (hemisphere sample points)
    VkBuffer       kernelBuffer = VK_NULL_HANDLE;
    VmaAllocation kernelMem    = VK_NULL_HANDLE;

    // Params UBO (proj, invProj, screenSize, radius, bias)
    VkBuffer       paramsBuffer = VK_NULL_HANDLE;
    VmaAllocation paramsMem    = VK_NULL_HANDLE;
    void*          paramsMapped = nullptr;

    // Samplers
    VkSampler      nearestSampler = VK_NULL_HANDLE;
    VkSampler      noiseSampler   = VK_NULL_HANDLE;

    // SSAO compute pass
    VkDescriptorSetLayout ssaoSetLayout      = VK_NULL_HANDLE;
    VkPipelineLayout      ssaoPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            ssaoPipeline       = VK_NULL_HANDLE;
    VkDescriptorPool      ssaoPool           = VK_NULL_HANDLE;
    VkDescriptorSet       ssaoSet            = VK_NULL_HANDLE;

    // Blur compute pass
    VkDescriptorSetLayout blurSetLayout      = VK_NULL_HANDLE;
    VkPipelineLayout      blurPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            blurPipeline       = VK_NULL_HANDLE;
    VkDescriptorPool      blurPool           = VK_NULL_HANDLE;
    VkDescriptorSet       blurSet            = VK_NULL_HANDLE;

    void createPrePassRenderPass(VulkanContext& ctx);
    void createPrePassPipeline(VulkanContext& ctx,
                               VkDescriptorSetLayout frameSetLayout);
    void createNoiseTexture(VulkanContext& ctx);
    void createKernelUBO(VulkanContext& ctx);
    void createParamsUBO(VulkanContext& ctx);
    void createSamplers(VulkanContext& ctx);
    void createComputePipelines(VulkanContext& ctx);
    void allocateDescriptorSets(VulkanContext& ctx);

    void createSizeDependentResources(VulkanContext& ctx);
    void destroySizeDependentResources(VulkanContext& ctx);
    void writeDescriptorSets(VulkanContext& ctx);
};
