#include "rhi/Image.h"
#include "rhi/VulkanContext.h"

#include <stdexcept>

// ================================================================
//  内部辅助：根据 old/new layout 推导 access mask 与 pipeline stage
// ================================================================
static void deduceBarrierMasks(VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkAccessFlags& srcAccess, VkAccessFlags& dstAccess,
                               VkPipelineStageFlags& srcStage, VkPipelineStageFlags& dstStage)
{
    srcAccess = 0;
    dstAccess = 0;

    switch (oldLayout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        srcAccess = 0;
        srcStage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        srcAccess = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        srcAccess = VK_ACCESS_SHADER_READ_BIT;
        srcStage  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        break;
    default:
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        break;
    }

    switch (newLayout) {
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        dstAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstStage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        dstAccess = VK_ACCESS_SHADER_READ_BIT;
        dstStage  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        dstAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dstStage  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    default:
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        break;
    }
}

// ================================================================
//  2D 图像
// ================================================================

void Image::createImage(VulkanContext& ctx,
                        uint32_t width, uint32_t height,
                        VkFormat format, VkImageTiling tiling,
                        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                        VkImage& image, VmaAllocation& allocation)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent        = {width, height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage         = VMA_MEMORY_USAGE_AUTO;
    allocCI.requiredFlags = properties;

    if (vmaCreateImage(ctx.allocator, &imageInfo, &allocCI,
                       &image, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("failed to create image!");
}


VkImageView Image::createImageView(VulkanContext& ctx,
                                   VkImage image, VkFormat format,
                                   VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};

    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView imageView;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture image view!");
    return imageView;
}

VkImageView Image::createLayeredImageView(VulkanContext& ctx,
                                          VkImage image, VkFormat format,
                                          uint32_t mipLevel, uint32_t layerCount)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = mipLevel;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = layerCount;

    VkImageView view;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("failed to create layered image view!");
    return view;
}

// ================================================================
//  Cubemap 图像
// ================================================================

void Image::createCubeImage(VulkanContext& ctx,
                            uint32_t size, uint32_t mipLevels,
                            VkFormat format, VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkImage& image, VmaAllocation& allocation)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent        = {size, size, 1};
    imageInfo.mipLevels     = mipLevels;
    imageInfo.arrayLayers   = 6;
    imageInfo.format        = format;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage         = VMA_MEMORY_USAGE_AUTO;
    allocCI.requiredFlags = properties;

    if (vmaCreateImage(ctx.allocator, &imageInfo, &allocCI,
                       &image, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("failed to create cube image!");
}

VkImageView Image::createCubeImageView(VulkanContext& ctx,
                                       VkImage image, VkFormat format,
                                       uint32_t mipLevelCount)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mipLevelCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 6;

    VkImageView view;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("failed to create cube image view!");
    return view;
}

VkImageView Image::createCubeImageViewSingleMip(VulkanContext& ctx,
                                                VkImage image, VkFormat format,
                                                uint32_t mipLevel)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = mipLevel;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 6;

    VkImageView view;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("failed to create cube image view (single mip)!");
    return view;
}

// ================================================================
//  布局转换
// ================================================================

void Image::transitionImageLayout(VulkanContext& ctx,
                                  VkImage image, VkFormat /*format*/,
                                  VkImageLayout oldLayout, VkImageLayout newLayout)
{
    transitionImageLayout(ctx, image, oldLayout, newLayout, 1, 1);
}

void Image::transitionImageLayout(VulkanContext& ctx,
                                  VkImage image,
                                  VkImageLayout oldLayout, VkImageLayout newLayout,
                                  uint32_t mipLevels, uint32_t layerCount)
{
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkAccessFlags srcAccess, dstAccess;
    VkPipelineStageFlags srcStage, dstStage;
    deduceBarrierMasks(oldLayout, newLayout, srcAccess, dstAccess, srcStage, dstStage);

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = oldLayout;
    barrier.newLayout                       = newLayout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = layerCount;
    barrier.srcAccessMask                   = srcAccess;
    barrier.dstAccessMask                   = dstAccess;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    ctx.endSingleTimeCommands(cmd);
}

// ================================================================
//  Buffer → Image 拷贝
// ================================================================

void Image::copyBufferToImage(VulkanContext& ctx,
                              VkBuffer buffer, VkImage image,
                              uint32_t width, uint32_t height)
{
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = {0, 0, 0};
    region.imageExtent                     = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    ctx.endSingleTimeCommands(cmd);
}
