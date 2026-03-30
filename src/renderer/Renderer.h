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

enum class RenderMode {
    Forward,
    DeferredMVP,
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
    RenderMode                   renderMode   = RenderMode::Forward;

    void init(VulkanContext& ctx, SwapChain& swapChain,
              Pipeline& pipeline, ShadowMap& shadowMap,
              IBLResources& ibl, SSAO& ssao,
              const std::vector<TextureManager*>& textures);
    void setRenderMode(RenderMode mode) { renderMode = mode; }
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
    std::vector<VkDescriptorSet> deferredDescriptorSets;

    VkSampler deferredSampler = VK_NULL_HANDLE;

    VkImage       gbufferAlbedoImage     = VK_NULL_HANDLE;
    VmaAllocation gbufferAlbedoAlloc     = VK_NULL_HANDLE;
    VkImageView   gbufferAlbedoView      = VK_NULL_HANDLE;

    VkImage       gbufferNormalImage     = VK_NULL_HANDLE;
    VmaAllocation gbufferNormalAlloc     = VK_NULL_HANDLE;
    VkImageView   gbufferNormalView      = VK_NULL_HANDLE;

    VkImage       gbufferPositionImage   = VK_NULL_HANDLE;
    VmaAllocation gbufferPositionAlloc   = VK_NULL_HANDLE;
    VkImageView   gbufferPositionView    = VK_NULL_HANDLE;

    VkImage       deferredDepthImage     = VK_NULL_HANDLE;
    VmaAllocation deferredDepthAlloc     = VK_NULL_HANDLE;
    VkImageView   deferredDepthView      = VK_NULL_HANDLE;

    VkRenderPass             deferredGeometryRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> deferredGeometryFramebuffers;
    VkPipelineLayout         deferredGeometryPipelineLayout = VK_NULL_HANDLE;
    VkPipeline               deferredGeometryPipeline = VK_NULL_HANDLE;

    VkRenderPass             deferredLightingRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> deferredLightingFramebuffers;
    VkDescriptorSetLayout    deferredSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout         deferredLightingPipelineLayout = VK_NULL_HANDLE;
    VkPipeline               deferredLightingPipeline = VK_NULL_HANDLE;
    VkPipelineLayout         deferredSkyboxPipelineLayout = VK_NULL_HANDLE;
    VkPipeline               deferredSkyboxPipeline = VK_NULL_HANDLE;

    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx, uint32_t numTextures);
    void createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                   ShadowMap& shadowMap, IBLResources& ibl,
                                   SSAO& ssao);
    void updateAODescriptorSets(VulkanContext& ctx, SSAO& ssao);
    void createMaterialDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                      const std::vector<TextureManager*>& textures);
    void createDeferredSampler(VulkanContext& ctx);
    void createDeferredDescriptorSets(VulkanContext& ctx);
    void createDeferredResources(VulkanContext& ctx, SwapChain& swapChain,
                                 Pipeline& pipeline);
    void destroyDeferredResources(VulkanContext& ctx);
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
    void recordForwardCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                                    SwapChain& swapChain, Pipeline& pipeline,
                                    ShadowMap& shadowMap, SSAO& ssao,
                                    const std::vector<RenderObject>& objects);
    void recordDeferredCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                                     SwapChain& swapChain, Pipeline& pipeline,
                                     ShadowMap& shadowMap, SSAO& ssao,
                                     const std::vector<RenderObject>& objects);
};
