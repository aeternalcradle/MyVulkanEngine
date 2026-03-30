#include "renderer/IBLResources.h"
#include "renderer/Pipeline.h"
#include "rhi/VulkanContext.h"
#include "rhi/Image.h"

#include <stb_image.h>

// tinyexr compression side references stbi_zlib_compress which stb_image.h
// does not provide. We only load EXR files, so a no-op stub is sufficient.
extern "C" unsigned char* stbi_zlib_compress(
    unsigned char* /*data*/, int /*data_len*/, int* out_len, int /*quality*/)
{
    if (out_len) *out_len = 0;
    return nullptr;
}

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

// ================================================================
//  ComputePass helper
// ================================================================

void IBLResources::ComputePass::destroy(VkDevice device) {
    if (pipeline)  vkDestroyPipeline(device, pipeline, nullptr);
    if (pipLayout) vkDestroyPipelineLayout(device, pipLayout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    if (pool)      vkDestroyDescriptorPool(device, pool, nullptr);
}

IBLResources::ComputePass IBLResources::createComputePass(
    VulkanContext& ctx,
    const std::string& spvPath,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t pushConstantSize)
{
    ComputePass cp{};

    // Descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &cp.setLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute descriptor set layout!");

    // Pipeline layout
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = pushConstantSize;

    VkPipelineLayoutCreateInfo pipLayoutInfo{};
    pipLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipLayoutInfo.setLayoutCount = 1;
    pipLayoutInfo.pSetLayouts    = &cp.setLayout;
    if (pushConstantSize > 0) {
        pipLayoutInfo.pushConstantRangeCount = 1;
        pipLayoutInfo.pPushConstantRanges    = &pushRange;
    }
    if (vkCreatePipelineLayout(ctx.device, &pipLayoutInfo, nullptr, &cp.pipLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout!");

    // Compute pipeline
    auto code = Pipeline::readFile(spvPath);
    VkShaderModule module = Pipeline::createShaderModule(ctx, code);

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module       = module;
    cpInfo.stage.pName        = "main";
    cpInfo.layout             = cp.pipLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &cp.pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline!");

    vkDestroyShaderModule(ctx.device, module, nullptr);

    // Descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& b : bindings) {
        poolSizes.push_back({b.descriptorType, b.descriptorCount});
    }
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &cp.pool) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute descriptor pool!");

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = cp.pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &cp.setLayout;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &cp.set) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate compute descriptor set!");

    return cp;
}

// ================================================================
//  Samplers
// ================================================================

void IBLResources::createSamplers(VulkanContext& ctx) {
    // Cubemap sampler (with mip levels for prefiltered map)
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter    = VK_FILTER_LINEAR;
    samplerInfo.minFilter    = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod       = static_cast<float>(PREFILTER_MIP_LEVELS);
    if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &cubemapSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create cubemap sampler!");

    // BRDF LUT sampler
    VkSamplerCreateInfo brdfInfo{};
    brdfInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    brdfInfo.magFilter    = VK_FILTER_LINEAR;
    brdfInfo.minFilter    = VK_FILTER_LINEAR;
    brdfInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    brdfInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    brdfInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    brdfInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    brdfInfo.maxLod       = 1.0f;
    if (vkCreateSampler(ctx.device, &brdfInfo, nullptr, &brdfLutSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create BRDF LUT sampler!");
}

// ================================================================
//  Load EXR equirect → env cubemap
// ================================================================

void IBLResources::loadEquirectAndConvert(
    VulkanContext& ctx, const std::string& exrPath,
    VkImage& envCube, VmaAllocation& envCubeMem, VkImageView& envCubeView)
{
    // --- Load EXR with tinyexr ---
    float* rgbaPixels = nullptr;
    int width = 0, height = 0;
    const char* err = nullptr;

    int ret = LoadEXR(&rgbaPixels, &width, &height, exrPath.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::string msg = "failed to load EXR: ";
        if (err) { msg += err; FreeEXRErrorMessage(err); }
        throw std::runtime_error(msg);
    }

    // --- Upload equirect as 2D RGBA32F image ---
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);

    VkBuffer       stagingBuffer;
    VmaAllocation  stagingMem;
    ctx.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMem);
    void* data;
    vmaMapMemory(ctx.allocator, stagingMem, &data);
    memcpy(data, rgbaPixels, static_cast<size_t>(imageSize));
    vmaUnmapMemory(ctx.allocator, stagingMem);
    free(rgbaPixels);

    VkImage        equirectImage;
    VmaAllocation  equirectMem;
    Image::createImage(ctx, static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                       VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       equirectImage, equirectMem);

    Image::transitionImageLayout(ctx, equirectImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1);
    Image::copyBufferToImage(ctx, stagingBuffer, equirectImage,
                             static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    Image::transitionImageLayout(ctx, equirectImage,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingMem);

    VkImageView equirectView = Image::createImageView(ctx, equirectImage,
                                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                                      VK_IMAGE_ASPECT_COLOR_BIT);

    // --- Create env cubemap ---
    Image::createCubeImage(ctx, ENV_CUBE_SIZE, 1, CUBE_FORMAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           envCube, envCubeMem);
    Image::transitionImageLayout(ctx, envCube,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 6);

    VkImageView envCubeArrayView = Image::createLayeredImageView(ctx, envCube, CUBE_FORMAT, 0, 6);

    // --- Compute pass: equirect → cube ---
    VkDescriptorSetLayoutBinding b0{};
    b0.binding         = 0;
    b0.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b0.descriptorCount = 1;
    b0.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding         = 1;
    b1.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b1.descriptorCount = 1;
    b1.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    auto cp = createComputePass(ctx, "shaders/equirect_to_cube_comp.spv", {b0, b1});

    VkDescriptorImageInfo equirectInfo{};
    equirectInfo.sampler     = cubemapSampler;
    equirectInfo.imageView   = equirectView;
    equirectInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo cubeStorageInfo{};
    cubeStorageInfo.imageView   = envCubeArrayView;
    cubeStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = cp.set;
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo      = &equirectInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = cp.set;
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &cubeStorageInfo;

    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipLayout, 0, 1, &cp.set, 0, nullptr);
    uint32_t gx = (ENV_CUBE_SIZE + 15) / 16;
    vkCmdDispatch(cmd, gx, gx, 6);
    ctx.endSingleTimeCommands(cmd);

    Image::transitionImageLayout(ctx, envCube,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 6);

    envCubeView = Image::createCubeImageView(ctx, envCube, CUBE_FORMAT, 1);

    // Cleanup temporaries
    vkDestroyImageView(ctx.device, envCubeArrayView, nullptr);
    cp.destroy(ctx.device);
    vkDestroyImageView(ctx.device, equirectView, nullptr);
    vmaDestroyImage(ctx.allocator, equirectImage, equirectMem);
}

// ================================================================
//  Irradiance convolution
// ================================================================

void IBLResources::computeIrradiance(VulkanContext& ctx, VkImageView envCubeView) {
    Image::createCubeImage(ctx, IRR_CUBE_SIZE, 1, CUBE_FORMAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           irradianceImage, irradianceMem);
    Image::transitionImageLayout(ctx, irradianceImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 6);

    VkImageView storageView = Image::createLayeredImageView(ctx, irradianceImage, CUBE_FORMAT, 0, 6);

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1; b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b1.descriptorCount = 1; b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    auto cp = createComputePass(ctx, "shaders/irradiance_comp.spv", {b0, b1});

    VkDescriptorImageInfo envInfo{};
    envInfo.sampler = cubemapSampler; envInfo.imageView = envCubeView;
    envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageView = storageView; storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cp.set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cp.set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storageInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipLayout, 0, 1, &cp.set, 0, nullptr);
    uint32_t gx = (IRR_CUBE_SIZE + 15) / 16;
    vkCmdDispatch(cmd, gx, gx, 6);
    ctx.endSingleTimeCommands(cmd);

    Image::transitionImageLayout(ctx, irradianceImage,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 6);
    irradianceView = Image::createCubeImageView(ctx, irradianceImage, CUBE_FORMAT, 1);

    vkDestroyImageView(ctx.device, storageView, nullptr);
    cp.destroy(ctx.device);
}

// ================================================================
//  Prefiltered environment map
// ================================================================

void IBLResources::computePrefilter(VulkanContext& ctx, VkImageView envCubeView) {
    Image::createCubeImage(ctx, PREFILTER_SIZE, PREFILTER_MIP_LEVELS, CUBE_FORMAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           prefilteredImage, prefilteredMem);
    Image::transitionImageLayout(ctx, prefilteredImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                 PREFILTER_MIP_LEVELS, 6);

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1; b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b1.descriptorCount = 1; b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    auto cp = createComputePass(ctx, "shaders/prefilter_comp.spv", {b0, b1},
                                static_cast<uint32_t>(sizeof(float)));

    for (uint32_t mip = 0; mip < PREFILTER_MIP_LEVELS; ++mip) {
        uint32_t mipSize = PREFILTER_SIZE >> mip;

        VkImageView mipStorageView = Image::createLayeredImageView(
            ctx, prefilteredImage, CUBE_FORMAT, mip, 6);

        VkDescriptorImageInfo envInfo{};
        envInfo.sampler = cubemapSampler; envInfo.imageView = envCubeView;
        envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo storageInfo{};
        storageInfo.imageView = mipStorageView; storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cp.set, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cp.set, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storageInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

        float roughness = static_cast<float>(mip) / static_cast<float>(PREFILTER_MIP_LEVELS - 1);

        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipLayout,
                                0, 1, &cp.set, 0, nullptr);
        vkCmdPushConstants(cmd, cp.pipLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &roughness);
        uint32_t gx = std::max(1u, (mipSize + 15) / 16);
        vkCmdDispatch(cmd, gx, gx, 6);
        ctx.endSingleTimeCommands(cmd);

        vkDestroyImageView(ctx.device, mipStorageView, nullptr);
    }

    Image::transitionImageLayout(ctx, prefilteredImage,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 PREFILTER_MIP_LEVELS, 6);
    prefilteredView = Image::createCubeImageView(ctx, prefilteredImage, CUBE_FORMAT, PREFILTER_MIP_LEVELS);

    cp.destroy(ctx.device);
}

// ================================================================
//  BRDF Integration LUT
// ================================================================

void IBLResources::computeBRDFLUT(VulkanContext& ctx) {
    Image::createImage(ctx, BRDF_LUT_SIZE, BRDF_LUT_SIZE,
                       BRDF_FORMAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       brdfLutImage, brdfLutMem);
    Image::transitionImageLayout(ctx, brdfLutImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 1);

    VkImageView storageView = Image::createImageView(ctx, brdfLutImage, BRDF_FORMAT,
                                                     VK_IMAGE_ASPECT_COLOR_BIT);

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    auto cp = createComputePass(ctx, "shaders/brdf_lut_comp.spv", {b0});

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageView = storageView; storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = cp.set;
    write.dstBinding      = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo      = &storageInfo;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipLayout,
                            0, 1, &cp.set, 0, nullptr);
    uint32_t gx = (BRDF_LUT_SIZE + 15) / 16;
    vkCmdDispatch(cmd, gx, gx, 1);
    ctx.endSingleTimeCommands(cmd);

    Image::transitionImageLayout(ctx, brdfLutImage,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);
    brdfLutView = Image::createImageView(ctx, brdfLutImage, BRDF_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);

    vkDestroyImageView(ctx.device, storageView, nullptr);
    cp.destroy(ctx.device);
}

// ================================================================
//  Public interface
// ================================================================

void IBLResources::create(VulkanContext& ctx, const std::string& exrPath) {
    createSamplers(ctx);

    loadEquirectAndConvert(ctx, exrPath, envCubeImage, envCubeMem, envCubeView);
    computeIrradiance(ctx, envCubeView);
    computePrefilter(ctx, envCubeView);
    computeBRDFLUT(ctx);
}

void IBLResources::destroy(VulkanContext& ctx) {
    vkDestroySampler(ctx.device, cubemapSampler, nullptr);
    vkDestroySampler(ctx.device, brdfLutSampler, nullptr);

    vkDestroyImageView(ctx.device, envCubeView, nullptr);
    vmaDestroyImage(ctx.allocator, envCubeImage, envCubeMem);

    vkDestroyImageView(ctx.device, irradianceView, nullptr);
    vmaDestroyImage(ctx.allocator, irradianceImage, irradianceMem);

    vkDestroyImageView(ctx.device, prefilteredView, nullptr);
    vmaDestroyImage(ctx.allocator, prefilteredImage, prefilteredMem);

    vkDestroyImageView(ctx.device, brdfLutView, nullptr);
    vmaDestroyImage(ctx.allocator, brdfLutImage, brdfLutMem);
}
