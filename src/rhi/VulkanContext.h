#pragma once

#include "rhi/VulkanTypes.h"
#include <vector>

// 管理 Vulkan 核心对象：Instance / Device / Surface / CommandPool
// 并提供内存、格式查询及缓冲区工具函数
class VulkanContext {
public:
    VkInstance               instance        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger  = VK_NULL_HANDLE;
    VkSurfaceKHR             surface         = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice  = VK_NULL_HANDLE;
    VkDevice                 device          = VK_NULL_HANDLE;
    VkQueue                  graphicsQueue   = VK_NULL_HANDLE;
    VkQueue                  presentQueue    = VK_NULL_HANDLE;
    VkCommandPool            commandPool     = VK_NULL_HANDLE;
    VmaAllocator             allocator       = VK_NULL_HANDLE;

    void init(GLFWwindow* window);
    void destroy();

    // ---- 供其他子系统调用的工具函数 ----
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) const;
    uint32_t           findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    VkFormat           findSupportedFormat(const std::vector<VkFormat>& candidates,
                                           VkImageTiling tiling,
                                           VkFormatFeatureFlags features) const;
    VkFormat           findDepthFormat() const;

    void            createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties,
                                 VkBuffer& buffer, VmaAllocation& allocation);
    void            copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void            endSingleTimeCommands(VkCommandBuffer commandBuffer);

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createAllocator();

    bool isDeviceSuitable(VkPhysicalDevice dev) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) const;
    bool checkValidationLayerSupport() const;
    std::vector<const char*> getRequiredExtensions() const;
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& ci);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};
