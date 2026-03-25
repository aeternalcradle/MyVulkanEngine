#pragma once

#include "rhi/VulkanTypes.h"
#include <string>

class VulkanContext;

// 负责纹理图像的加载、ImageView 创建和 Sampler 创建
class TextureManager {
public:
    VkImage        textureImage       = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory = VK_NULL_HANDLE;
    VkImageView    textureImageView   = VK_NULL_HANDLE;
    VkSampler      textureSampler     = VK_NULL_HANDLE;

    void loadTexture(VulkanContext& ctx, const std::string& path);
    void createSolidColor(VulkanContext& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    void destroy(VulkanContext& ctx);

private:
    void createTextureImage(VulkanContext& ctx, const std::string& path);
    void createTextureImageView(VulkanContext& ctx);
    void createTextureSampler(VulkanContext& ctx);
};
