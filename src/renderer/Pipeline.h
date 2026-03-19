#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>
#include <string>

class VulkanContext;
class SwapChain;

// 管理 RenderPass、DescriptorSetLayout 和 GraphicsPipeline
class Pipeline {
public:
    VkRenderPass          renderPass          = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            graphicsPipeline    = VK_NULL_HANDLE;

    void create(VulkanContext& ctx, SwapChain& swapChain);
    void destroy(VulkanContext& ctx);

private:
    void createRenderPass(VulkanContext& ctx, VkFormat swapChainImageFormat);
    void createDescriptorSetLayout(VulkanContext& ctx);
    void createGraphicsPipeline(VulkanContext& ctx, VkExtent2D swapChainExtent);

    VkShaderModule        createShaderModule(VulkanContext& ctx, const std::vector<char>& code);
    static std::vector<char> readFile(const std::string& filename);
};
