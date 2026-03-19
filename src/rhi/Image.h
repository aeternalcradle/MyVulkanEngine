#pragma once

#include "rhi/VulkanTypes.h"

class VulkanContext;

// 图像创建、视图创建、布局转换和 Buffer→Image 拷贝工具类（纯静态）
class Image {
public:
    static void createImage(VulkanContext& ctx,
                            uint32_t width, uint32_t height,
                            VkFormat format, VkImageTiling tiling,
                            VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkImage& image, VkDeviceMemory& imageMemory);

    static VkImageView createImageView(VulkanContext& ctx,
                                       VkImage image,
                                       VkFormat format,
                                       VkImageAspectFlags aspectFlags);

    static void transitionImageLayout(VulkanContext& ctx,
                                      VkImage image, VkFormat format,
                                      VkImageLayout oldLayout, VkImageLayout newLayout);

    static void copyBufferToImage(VulkanContext& ctx,
                                  VkBuffer buffer, VkImage image,
                                  uint32_t width, uint32_t height);
};
