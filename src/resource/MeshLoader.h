#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>
#include <string>

class VulkanContext;

// 负责 OBJ 模型加载及顶点/索引缓冲区创建
class MeshLoader {
public:
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
    VmaAllocation vertexBufferAlloc = VK_NULL_HANDLE;
    VkBuffer       indexBuffer        = VK_NULL_HANDLE;
    VmaAllocation indexBufferAlloc  = VK_NULL_HANDLE;

    void loadMesh(VulkanContext& ctx, const std::string& modelPath);
    void createPlane(VulkanContext& ctx, float size = 10.0f);
    void createSphere(VulkanContext& ctx, float radius = 0.5f,
                      uint32_t sectors = 64, uint32_t stacks = 32);
    void destroy(VulkanContext& ctx);

private:
    void loadModel(const std::string& modelPath);
    void computeSmoothNormals();
    void createVertexBuffer(VulkanContext& ctx);
    void createIndexBuffer(VulkanContext& ctx);
};
