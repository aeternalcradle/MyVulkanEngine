#pragma once

#include "rhi/VulkanTypes.h"

class VulkanContext;

// 图像创建、视图创建、布局转换和 Buffer→Image 拷贝工具类（纯静态）
class Image {
public:
    // ---- 2D 图像 ----
    static void createImage(VulkanContext& ctx,
                            uint32_t width, uint32_t height,
                            VkFormat format, VkImageTiling tiling,
                            VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkImage& image, VmaAllocation& allocation);

    static VkImageView createImageView(VulkanContext& ctx,
                                       VkImage image,
                                       VkFormat format,
                                       VkImageAspectFlags aspectFlags);

    // 2D array view: compute shader 写入 cubemap 的指定 mip level（6 layers）
    static VkImageView createLayeredImageView(VulkanContext& ctx,
                                              VkImage image,
                                              VkFormat format,
                                              uint32_t mipLevel,
                                              uint32_t layerCount);

    // ---- Cubemap ----
    static void createCubeImage(VulkanContext& ctx,
                                uint32_t size, uint32_t mipLevels,
                                VkFormat format, VkImageUsageFlags usage,
                                VkMemoryPropertyFlags properties,
                                VkImage& image, VmaAllocation& allocation);

    static VkImageView createCubeImageView(VulkanContext& ctx,
                                           VkImage image,
                                           VkFormat format,
                                           uint32_t mipLevels);

    static VkImageView createCubeImageViewSingleMip(VulkanContext& ctx,
                                                    VkImage image,
                                                    VkFormat format,
                                                    uint32_t mipLevel);

    // ---- 布局转换（通用版本，支持 cubemap / mip） ----
    static void transitionImageLayout(VulkanContext& ctx,
                                      VkImage image, VkFormat format,
                                      VkImageLayout oldLayout, VkImageLayout newLayout);

    static void transitionImageLayout(VulkanContext& ctx,
                                      VkImage image,
                                      VkImageLayout oldLayout, VkImageLayout newLayout,
                                      uint32_t mipLevels, uint32_t layerCount);

    // ---- 拷贝 ----
    static void copyBufferToImage(VulkanContext& ctx,
                                  VkBuffer buffer, VkImage image,
                                  uint32_t width, uint32_t height);
};
