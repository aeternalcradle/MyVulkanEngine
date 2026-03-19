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
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer       indexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;

    void loadMesh(VulkanContext& ctx, const std::string& modelPath);
    void destroy(VulkanContext& ctx);

private:
    void loadModel(const std::string& modelPath);
    void createVertexBuffer(VulkanContext& ctx);
    void createIndexBuffer(VulkanContext& ctx);
};
