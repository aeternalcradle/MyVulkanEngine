#include "rhi/SwapChain.h"
#include "rhi/VulkanContext.h"
#include "rhi/Image.h"

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <array>

void SwapChain::create(VulkanContext& ctx, GLFWwindow* window) {
    createSwapChain(ctx, window);
    createImageViews(ctx);
    createDepthResources(ctx);
}

void SwapChain::createFramebuffers(VulkanContext& ctx, VkRenderPass renderPass) {
    framebuffers.resize(imageViews.size());
    for (size_t i = 0; i < imageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = { imageViews[i], depthImageView };

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = renderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments    = attachments.data();
        fbInfo.width           = extent.width;
        fbInfo.height          = extent.height;
        fbInfo.layers          = 1;

        if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create framebuffer!");
    }
}

void SwapChain::cleanup(VulkanContext& ctx) {
    vkDestroyImageView(ctx.device, depthImageView, nullptr);
    vmaDestroyImage(ctx.allocator, depthImage, depthImageAlloc);

    for (auto fb : framebuffers)   vkDestroyFramebuffer(ctx.device, fb, nullptr);
    for (auto iv : imageViews)     vkDestroyImageView(ctx.device, iv, nullptr);
    framebuffers.clear();
    imageViews.clear();

    vkDestroySwapchainKHR(ctx.device, swapChain, nullptr);
}

void SwapChain::destroy(VulkanContext& ctx) {
    cleanup(ctx);
}

SwapChainSupportDetails SwapChain::querySupport(VkPhysicalDevice dev, VkSurfaceKHR surf) const {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surf, &details.capabilities);

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &fmtCount, nullptr);
    if (fmtCount) {
        details.formats.resize(fmtCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surf, &fmtCount, details.formats.data());
    }

    uint32_t pmCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surf, &pmCount, nullptr);
    if (pmCount) {
        details.presentModes.resize(pmCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surf, &pmCount, details.presentModes.data());
    }
    return details;
}

void SwapChain::createSwapChain(VulkanContext& ctx, GLFWwindow* window) {
    SwapChainSupportDetails support = querySupport(ctx.physicalDevice, ctx.surface);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR   presentMode   = chooseSwapPresentMode(support.presentModes);
    VkExtent2D         swapExtent    = chooseSwapExtent(support.capabilities, window);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = ctx.surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = surfaceFormat.format;
    ci.imageColorSpace  = surfaceFormat.colorSpace;
    ci.imageExtent      = swapExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = ctx.findQueueFamilies(ctx.physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform   = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = presentMode;
    ci.clipped        = VK_TRUE;

    if (vkCreateSwapchainKHR(ctx.device, &ci, nullptr, &swapChain) != VK_SUCCESS)
        throw std::runtime_error("failed to create swap chain!");

    vkGetSwapchainImagesKHR(ctx.device, swapChain, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(ctx.device, swapChain, &imageCount, images.data());

    imageFormat = surfaceFormat.format;
    extent      = swapExtent;
}

void SwapChain::createImageViews(VulkanContext& ctx) {
    imageViews.resize(images.size());
    for (uint32_t i = 0; i < images.size(); i++)
        imageViews[i] = Image::createImageView(ctx, images[i], imageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
}

void SwapChain::createDepthResources(VulkanContext& ctx) {
    VkFormat depthFormat = ctx.findDepthFormat();
    Image::createImage(ctx, extent.width, extent.height, depthFormat,
                       VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageAlloc);
    depthImageView = Image::createImageView(ctx, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats[0];
}

VkPresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes) const {
    for (const auto& m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& cap, GLFWwindow* window) const {
    if (cap.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return cap.currentExtent;

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    VkExtent2D actual = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    actual.width  = std::clamp(actual.width,  cap.minImageExtent.width,  cap.maxImageExtent.width);
    actual.height = std::clamp(actual.height, cap.minImageExtent.height, cap.maxImageExtent.height);
    return actual;
}
