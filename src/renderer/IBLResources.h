#pragma once

#include "rhi/VulkanTypes.h"
#include <string>

class VulkanContext;

class IBLResources {
public:
    static constexpr uint32_t ENV_CUBE_SIZE        = 512;
    static constexpr uint32_t IRR_CUBE_SIZE        = 32;
    static constexpr uint32_t PREFILTER_SIZE       = 128;
    static constexpr uint32_t PREFILTER_MIP_LEVELS = 5;
    static constexpr uint32_t BRDF_LUT_SIZE        = 512;
    static constexpr VkFormat CUBE_FORMAT           = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat BRDF_FORMAT           = VK_FORMAT_R16G16_SFLOAT;

    VkImage        irradianceImage  = VK_NULL_HANDLE;
    VmaAllocation  irradianceMem    = VK_NULL_HANDLE;
    VkImageView    irradianceView   = VK_NULL_HANDLE;

    VkImage        prefilteredImage = VK_NULL_HANDLE;
    VmaAllocation  prefilteredMem   = VK_NULL_HANDLE;
    VkImageView    prefilteredView  = VK_NULL_HANDLE;

    VkImage        brdfLutImage     = VK_NULL_HANDLE;
    VmaAllocation  brdfLutMem       = VK_NULL_HANDLE;
    VkImageView    brdfLutView      = VK_NULL_HANDLE;

    VkSampler      cubemapSampler   = VK_NULL_HANDLE;
    VkSampler      brdfLutSampler   = VK_NULL_HANDLE;

    void create(VulkanContext& ctx, const std::string& exrPath);
    void destroy(VulkanContext& ctx);

private:
    void createSamplers(VulkanContext& ctx);

    void loadEquirectAndConvert(VulkanContext& ctx, const std::string& exrPath,
                                VkImage& envCube, VmaAllocation& envCubeMem,
                                VkImageView& envCubeView);
    void computeIrradiance(VulkanContext& ctx, VkImageView envCubeView);
    void computePrefilter(VulkanContext& ctx, VkImageView envCubeView);
    void computeBRDFLUT(VulkanContext& ctx);

    struct ComputePass {
        VkDescriptorSetLayout setLayout   = VK_NULL_HANDLE;
        VkPipelineLayout      pipLayout   = VK_NULL_HANDLE;
        VkPipeline            pipeline    = VK_NULL_HANDLE;
        VkDescriptorPool      pool        = VK_NULL_HANDLE;
        VkDescriptorSet       set         = VK_NULL_HANDLE;

        void destroy(VkDevice device);
    };

    ComputePass createComputePass(
        VulkanContext& ctx,
        const std::string& spvPath,
        const std::vector<VkDescriptorSetLayoutBinding>& bindings,
        uint32_t pushConstantSize = 0);
};
