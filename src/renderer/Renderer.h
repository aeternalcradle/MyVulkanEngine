#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;
class SwapChain;
class Pipeline;
class ShadowMap;
class IBLResources;
class SSAO;
class TextureManager;
class MeshLoader;
class Window;

struct RenderObject {
    MeshLoader*  mesh;
    uint32_t     textureIndex;
    glm::mat4    transform;
    float        metallic  = 0.0f;
    float        roughness = 0.5f;
};

class Renderer {
public:
    std::vector<VkBuffer>        uniformBuffers;
    std::vector<VmaAllocation>   uniformBuffersAlloc;
    std::vector<void*>           uniformBuffersMapped;

    VkDescriptorPool             descriptorPool = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore>     imageAvailableSemaphores;
    std::vector<VkSemaphore>     renderFinishedSemaphores;
    std::vector<VkFence>         inFlightFences;
    uint32_t                     currentFrame = 0;

    void init(VulkanContext& ctx, SwapChain& swapChain,
              Pipeline& pipeline, ShadowMap& shadowMap,
              IBLResources& ibl, SSAO& ssao,
              const std::vector<TextureManager*>& textures);
    void destroy(VulkanContext& ctx);

    void drawFrame(VulkanContext& ctx, Window& window,
                   SwapChain& swapChain, Pipeline& pipeline,
                   ShadowMap& shadowMap,
                   SSAO& ssao,
                   const std::vector<RenderObject>& objects,
                   const glm::vec3& cameraPos,
                   const glm::vec3& cameraTarget,
                   float farPlane = 10.0f);

private:
    std::vector<VkDescriptorSet> frameDescriptorSets;
    std::vector<std::vector<VkDescriptorSet>> materialDescriptorSets;

    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx, uint32_t numTextures);
    void createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                   ShadowMap& shadowMap, IBLResources& ibl,
                                   SSAO& ssao);
    void updateAODescriptorSets(VulkanContext& ctx, SSAO& ssao);
    void createMaterialDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                      const std::vector<TextureManager*>& textures);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx, uint32_t swapChainImageCount);

    void updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage,
                             const glm::vec3& cameraPos,
                             const glm::vec3& cameraTarget,
                             float farPlane,
                             SSAO& ssao);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                             SwapChain& swapChain, Pipeline& pipeline,
                             ShadowMap& shadowMap,
                             SSAO& ssao,
                             const std::vector<RenderObject>& objects);
};
