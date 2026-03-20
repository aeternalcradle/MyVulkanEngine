#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;
class SwapChain;
class Pipeline;
class TextureManager;
class MeshLoader;
class Window;

// 每个需要绘制的物体：网格 + 纹理索引 + 模型矩阵（每帧传入）
struct RenderObject {
    MeshLoader*  mesh;
    uint32_t     textureIndex;  // 对应 init() 传入的 textures 数组下标
    glm::mat4    transform;
};

// 管理帧同步、命令缓冲、UBO、描述符集及主渲染循环
class Renderer {
public:
    std::vector<VkBuffer>        uniformBuffers;
    std::vector<VkDeviceMemory>  uniformBuffersMemory;
    std::vector<void*>           uniformBuffersMapped;

    VkDescriptorPool             descriptorPool = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore>     imageAvailableSemaphores;
    std::vector<VkSemaphore>     renderFinishedSemaphores;
    std::vector<VkFence>         inFlightFences;
    uint32_t                     currentFrame = 0;

    // textures[i] 对应 RenderObject::textureIndex == i
    void init(VulkanContext& ctx, SwapChain& swapChain,
              Pipeline& pipeline,
              const std::vector<TextureManager*>& textures);
    void destroy(VulkanContext& ctx);

    void drawFrame(VulkanContext& ctx, Window& window,
                   SwapChain& swapChain, Pipeline& pipeline,
                   const std::vector<RenderObject>& objects);

private:
    // [textureIndex][frameIndex]
    std::vector<std::vector<VkDescriptorSet>> objectDescriptorSets;

    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx, uint32_t numTextures);
    void createDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                              const std::vector<TextureManager*>& textures);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);

    void updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                             SwapChain& swapChain, Pipeline& pipeline,
                             const std::vector<RenderObject>& objects);
};
