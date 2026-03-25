#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;
class SwapChain;
class Pipeline;
class ShadowMap;
class TextureManager;
class MeshLoader;
class Window;

struct RenderObject {
    MeshLoader*  mesh;
    uint32_t     textureIndex;
    glm::mat4    transform;
};

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

    void init(VulkanContext& ctx, SwapChain& swapChain,
              Pipeline& pipeline, ShadowMap& shadowMap,
              const std::vector<TextureManager*>& textures);
    void destroy(VulkanContext& ctx);

    void drawFrame(VulkanContext& ctx, Window& window,
                   SwapChain& swapChain, Pipeline& pipeline,
                   ShadowMap& shadowMap,
                   const std::vector<RenderObject>& objects);

private:
    // set 0: per-frame (UBO + shadow map) [frameIndex]
    std::vector<VkDescriptorSet> frameDescriptorSets;
    // set 1: per-material (diffuse texture) [textureIndex][frameIndex]
    std::vector<std::vector<VkDescriptorSet>> materialDescriptorSets;

    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx, uint32_t numTextures);
    void createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline, ShadowMap& shadowMap);
    void createMaterialDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                      const std::vector<TextureManager*>& textures);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);

    void updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                             SwapChain& swapChain, Pipeline& pipeline,
                             ShadowMap& shadowMap,
                             const std::vector<RenderObject>& objects);
};
