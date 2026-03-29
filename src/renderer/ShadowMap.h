#pragma once

#include "rhi/VulkanTypes.h"

class VulkanContext;

class ShadowMap {
public:
    static constexpr uint32_t SHADOW_MAP_RESOLUTION = 2048;

    VkImage          depthImage          = VK_NULL_HANDLE;
    VmaAllocation    depthImageAlloc     = VK_NULL_HANDLE;
    VkImageView      depthImageView      = VK_NULL_HANDLE;
    VkSampler        sampler             = VK_NULL_HANDLE;
    VkRenderPass     renderPass          = VK_NULL_HANDLE;
    VkFramebuffer    framebuffer         = VK_NULL_HANDLE;
    VkFormat         depthFormat         = VK_FORMAT_UNDEFINED;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       shadowPipeline       = VK_NULL_HANDLE;

    void create(VulkanContext& ctx, VkDescriptorSetLayout frameSetLayout);
    void destroy(VulkanContext& ctx);

private:
    void createDepthResources(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createSampler(VulkanContext& ctx);
    void createPipeline(VulkanContext& ctx, VkDescriptorSetLayout frameSetLayout);
};
