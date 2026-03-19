#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;
class SwapChain;
class Pipeline;
class TextureManager;
class MeshLoader;
class Window;

// 管理帧同步、命令缓冲、UBO、描述符集及主渲染循环
class Renderer {
public:
    std::vector<VkBuffer>        uniformBuffers;
    std::vector<VkDeviceMemory>  uniformBuffersMemory;
    std::vector<void*>           uniformBuffersMapped;

    VkDescriptorPool             descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore>     imageAvailableSemaphores;
    std::vector<VkSemaphore>     renderFinishedSemaphores;
    std::vector<VkFence>         inFlightFences;
    uint32_t                     currentFrame = 0;

    void init(VulkanContext& ctx, SwapChain& swapChain,
              Pipeline& pipeline, TextureManager& textureMgr);
    void destroy(VulkanContext& ctx);

    void drawFrame(VulkanContext& ctx, Window& window,
                   SwapChain& swapChain, Pipeline& pipeline, MeshLoader& mesh);

private:
    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx, Pipeline& pipeline, TextureManager& textureMgr);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);

    void updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                             SwapChain& swapChain, Pipeline& pipeline, MeshLoader& mesh);
};
