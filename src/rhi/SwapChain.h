#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

class VulkanContext;

// 管理 SwapChain、ImageView、深度缓冲和 Framebuffer
class SwapChain {
public:
    VkSwapchainKHR             swapChain        = VK_NULL_HANDLE;
    std::vector<VkImage>       images;
    VkFormat                   imageFormat      = VK_FORMAT_UNDEFINED;
    VkExtent2D                 extent           = {0, 0};
    std::vector<VkImageView>   imageViews;
    std::vector<VkFramebuffer> framebuffers;

    VkImage        depthImage       = VK_NULL_HANDLE;
    VmaAllocation depthImageAlloc = VK_NULL_HANDLE;
    VkImageView    depthImageView   = VK_NULL_HANDLE;

    void create(VulkanContext& ctx, GLFWwindow* window);
    void createFramebuffers(VulkanContext& ctx, VkRenderPass renderPass);
    void cleanup(VulkanContext& ctx);   // 仅销毁可重建部分（不含 swapChain 本身之外的持久对象）
    void destroy(VulkanContext& ctx);   // 完整销毁

    SwapChainSupportDetails querySupport(VkPhysicalDevice device, VkSurfaceKHR surface) const;

private:
    void createSwapChain(VulkanContext& ctx, GLFWwindow* window);
    void createImageViews(VulkanContext& ctx);
    void createDepthResources(VulkanContext& ctx);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
    VkPresentModeKHR   chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes) const;
    VkExtent2D         chooseSwapExtent(const VkSurfaceCapabilitiesKHR& cap, GLFWwindow* window) const;
};
