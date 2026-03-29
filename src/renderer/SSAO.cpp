#include "renderer/SSAO.h"
#include "renderer/Pipeline.h"
#include "renderer/Renderer.h"
#include "rhi/VulkanContext.h"
#include "rhi/Image.h"
#include "resource/MeshLoader.h"

#include <glm/gtc/packing.hpp>

#include <stdexcept>
#include <random>
#include <cstring>
#include <array>

// Matches std140 layout in ssao.comp (binding 4)
struct SSAOParams {
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 invProj;
    alignas(8)  glm::vec2 screenSize;
    float radius;
    float bias;
};

// ================================================================
//  Public interface
// ================================================================

void SSAO::create(VulkanContext& ctx, Pipeline& pipeline,
                  uint32_t w, uint32_t h)
{
    width_       = w;
    height_      = h;
    depthFormat_ = ctx.findDepthFormat();

    createPrePassRenderPass(ctx);
    createPrePassPipeline(ctx, pipeline.frameSetLayout);
    createNoiseTexture(ctx);
    createKernelUBO(ctx);
    createParamsUBO(ctx);
    createSamplers(ctx);
    createComputePipelines(ctx);
    allocateDescriptorSets(ctx);

    createSizeDependentResources(ctx);
    writeDescriptorSets(ctx);
}

void SSAO::destroy(VulkanContext& ctx) {
    destroySizeDependentResources(ctx);

    vkDestroyPipeline(ctx.device, blurPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, blurPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, blurSetLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, blurPool, nullptr);

    vkDestroyPipeline(ctx.device, ssaoPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, ssaoPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, ssaoSetLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, ssaoPool, nullptr);

    vkDestroySampler(ctx.device, aoSampler, nullptr);
    vkDestroySampler(ctx.device, noiseSampler, nullptr);
    vkDestroySampler(ctx.device, nearestSampler, nullptr);

    vmaUnmapMemory(ctx.allocator, paramsMem);
    vmaDestroyBuffer(ctx.allocator, paramsBuffer, paramsMem);

    vmaDestroyBuffer(ctx.allocator, kernelBuffer, kernelMem);

    vkDestroyImageView(ctx.device, noiseView, nullptr);
    vmaDestroyImage(ctx.allocator, noiseImage, noiseMem);

    vkDestroyPipeline(ctx.device, prePassPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, prePassPipelineLayout, nullptr);
    vkDestroyRenderPass(ctx.device, prePassRenderPass, nullptr);
}

void SSAO::resize(VulkanContext& ctx, uint32_t w, uint32_t h) {
    destroySizeDependentResources(ctx);
    width_  = w;
    height_ = h;
    createSizeDependentResources(ctx);
    writeDescriptorSets(ctx);
}

void SSAO::updateParams(const glm::mat4& proj, const glm::vec2& screenSize) {
    SSAOParams p{};
    p.proj       = proj;
    p.invProj    = glm::inverse(proj);
    p.screenSize = screenSize;
    p.radius     = DEFAULT_RADIUS;
    p.bias       = DEFAULT_BIAS;
    memcpy(paramsMapped, &p, sizeof(p));
}

// ================================================================
//  Command recording
// ================================================================

void SSAO::recordPrePass(VkCommandBuffer cmd, VkDescriptorSet frameDescSet,
                         const std::vector<RenderObject>& objects)
{
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass        = prePassRenderPass;
    rpInfo.framebuffer       = prePassFramebuffer;
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = {width_, height_};
    rpInfo.clearValueCount   = static_cast<uint32_t>(clearValues.size());
    rpInfo.pClearValues      = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, prePassPipeline);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(width_);
    viewport.height   = static_cast<float>(height_);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, {width_, height_}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            prePassPipelineLayout, 0, 1,
                            &frameDescSet, 0, nullptr);

    for (const auto& obj : objects) {
        VkBuffer     vertexBuffers[] = { obj.mesh->vertexBuffer };
        VkDeviceSize offsets[]       = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, obj.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        PushConstants pc{ obj.transform };
        vkCmdPushConstants(cmd, prePassPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(PushConstants), &pc);

        vkCmdDrawIndexed(cmd,
                         static_cast<uint32_t>(obj.mesh->indices.size()),
                         1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

void SSAO::recordCompute(VkCommandBuffer cmd) {
    // Transition blurredAO: SHADER_READ_ONLY → GENERAL (prepare for blur write)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = blurredAOImage;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // ---- SSAO Compute ----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ssaoPipelineLayout, 0, 1, &ssaoSet, 0, nullptr);

    uint32_t gx = (width_  + 15) / 16;
    uint32_t gy = (height_ + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Barrier: SSAO compute write → blur compute read (rawAO stays in GENERAL)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = rawAOImage;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // ---- Blur Compute ----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            blurPipelineLayout, 0, 1, &blurSet, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition blurredAO: GENERAL → SHADER_READ_ONLY (for main pass)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = blurredAOImage;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}

// ================================================================
//  Pre-pass render pass
// ================================================================

void SSAO::createPrePassRenderPass(VulkanContext& ctx) {
    // Attachment 0: view-space normals
    VkAttachmentDescription normalAttachment{};
    normalAttachment.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    normalAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    normalAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    normalAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    normalAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    normalAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    normalAttachment.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Attachment 1: depth
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = depthFormat_;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference normalRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &normalRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};

    deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass      = 0;
    deps[0].srcStageMask    = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    deps[1].srcSubpass      = 0;
    deps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask    = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {
        normalAttachment, depthAttachment
    };

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments    = attachments.data();
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies   = deps;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &prePassRenderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO pre-pass render pass!");
}

// ================================================================
//  Pre-pass graphics pipeline
// ================================================================

void SSAO::createPrePassPipeline(VulkanContext& ctx,
                                 VkDescriptorSetLayout frameSetLayout)
{
    auto vertCode = Pipeline::readFile("shaders/ssao_prepass_vert.spv");
    auto fragCode = Pipeline::readFile("shaders/ssao_prepass_frag.spv");
    VkShaderModule vertModule = Pipeline::createShaderModule(ctx, vertCode);
    VkShaderModule fragModule = Pipeline::createShaderModule(ctx, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    auto bindDesc  = Vertex::getBindingDescription();
    auto attrDescs = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &frameSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr,
                               &prePassPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO pre-pass pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = prePassPipelineLayout;
    pipelineInfo.renderPass          = prePassRenderPass;
    pipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr,
                                  &prePassPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO pre-pass pipeline!");

    vkDestroyShaderModule(ctx.device, fragModule, nullptr);
    vkDestroyShaderModule(ctx.device, vertModule, nullptr);
}

// ================================================================
//  4x4 noise texture (random tangent-space rotation vectors)
// ================================================================

void SSAO::createNoiseTexture(VulkanContext& ctx) {
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    struct HalfVec4 { uint16_t x, y, z, w; };
    HalfVec4 noiseData[NOISE_DIM * NOISE_DIM];

    for (int i = 0; i < NOISE_DIM * NOISE_DIM; ++i) {
        glm::vec2 v = glm::normalize(glm::vec2(dist(gen), dist(gen)));
        noiseData[i].x = glm::packHalf1x16(v.x);
        noiseData[i].y = glm::packHalf1x16(v.y);
        noiseData[i].z = glm::packHalf1x16(0.0f);
        noiseData[i].w = glm::packHalf1x16(0.0f);
    }

    VkDeviceSize imageSize = sizeof(noiseData);

    VkBuffer stagingBuffer;
    VmaAllocation stagingMem;
    ctx.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMem);

    void* data;
    vmaMapMemory(ctx.allocator, stagingMem, &data);
    memcpy(data, noiseData, imageSize);
    vmaUnmapMemory(ctx.allocator, stagingMem);

    Image::createImage(ctx, NOISE_DIM, NOISE_DIM,
                       VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       noiseImage, noiseMem);

    Image::transitionImageLayout(ctx, noiseImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1);
    Image::copyBufferToImage(ctx, stagingBuffer, noiseImage, NOISE_DIM, NOISE_DIM);
    Image::transitionImageLayout(ctx, noiseImage,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingMem);

    noiseView = Image::createImageView(ctx, noiseImage,
                                       VK_FORMAT_R16G16B16A16_SFLOAT,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
}

// ================================================================
//  Hemisphere kernel UBO (64 samples, weighted towards origin)
// ================================================================

void SSAO::createKernelUBO(VulkanContext& ctx) {
    std::default_random_engine gen(0);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distNeg(-1.0f, 1.0f);

    glm::vec4 samples[KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        glm::vec3 s(distNeg(gen), distNeg(gen), dist01(gen));
        s = glm::normalize(s) * dist01(gen);

        float scale = static_cast<float>(i) / static_cast<float>(KERNEL_SIZE);
        scale = 0.1f + scale * scale * 0.9f;
        s *= scale;

        samples[i] = glm::vec4(s, 0.0f);
    }

    VkDeviceSize bufSize = sizeof(samples);

    VkBuffer stagingBuffer;
    VmaAllocation stagingMem;
    ctx.createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMem);

    void* data;
    vmaMapMemory(ctx.allocator, stagingMem, &data);
    memcpy(data, samples, bufSize);
    vmaUnmapMemory(ctx.allocator, stagingMem);

    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     kernelBuffer, kernelMem);
    ctx.copyBuffer(stagingBuffer, kernelBuffer, bufSize);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingMem);
}

// ================================================================
//  Params UBO (host-visible, updated every frame)
// ================================================================

void SSAO::createParamsUBO(VulkanContext& ctx) {
    VkDeviceSize bufSize = sizeof(SSAOParams);
    ctx.createBuffer(bufSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     paramsBuffer, paramsMem);
    vmaMapMemory(ctx.allocator, paramsMem, &paramsMapped);
}

// ================================================================
//  Samplers
// ================================================================

void SSAO::createSamplers(VulkanContext& ctx) {
    VkSamplerCreateInfo nearestInfo{};
    nearestInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    nearestInfo.magFilter    = VK_FILTER_NEAREST;
    nearestInfo.minFilter    = VK_FILTER_NEAREST;
    nearestInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    nearestInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    nearestInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    nearestInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    nearestInfo.maxLod       = 1.0f;
    if (vkCreateSampler(ctx.device, &nearestInfo, nullptr, &nearestSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO nearest sampler!");

    VkSamplerCreateInfo noiseInfo{};
    noiseInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    noiseInfo.magFilter    = VK_FILTER_NEAREST;
    noiseInfo.minFilter    = VK_FILTER_NEAREST;
    noiseInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    noiseInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    noiseInfo.maxLod       = 1.0f;
    if (vkCreateSampler(ctx.device, &noiseInfo, nullptr, &noiseSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO noise sampler!");

    VkSamplerCreateInfo aoInfo{};
    aoInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    aoInfo.magFilter    = VK_FILTER_LINEAR;
    aoInfo.minFilter    = VK_FILTER_LINEAR;
    aoInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    aoInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    aoInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    aoInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    aoInfo.maxLod       = 1.0f;
    if (vkCreateSampler(ctx.device, &aoInfo, nullptr, &aoSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO AO sampler!");
}

// ================================================================
//  Compute pipelines (SSAO + blur)
// ================================================================

void SSAO::createComputePipelines(VulkanContext& ctx) {
    // ---- SSAO compute ----
    {
        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = 6;
        layoutCI.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device, &layoutCI, nullptr,
                                        &ssaoSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create SSAO compute set layout!");

        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 1;
        plCI.pSetLayouts    = &ssaoSetLayout;
        if (vkCreatePipelineLayout(ctx.device, &plCI, nullptr,
                                   &ssaoPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create SSAO compute pipeline layout!");

        auto code = Pipeline::readFile("shaders/ssao_comp.spv");
        VkShaderModule module = Pipeline::createShaderModule(ctx, code);

        VkComputePipelineCreateInfo cpCI{};
        cpCI.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpCI.stage.module = module;
        cpCI.stage.pName  = "main";
        cpCI.layout       = ssaoPipelineLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpCI,
                                     nullptr, &ssaoPipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create SSAO compute pipeline!");

        vkDestroyShaderModule(ctx.device, module, nullptr);

        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1}
        };
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = 3;
        poolCI.pPoolSizes    = poolSizes;
        if (vkCreateDescriptorPool(ctx.device, &poolCI, nullptr,
                                   &ssaoPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create SSAO descriptor pool!");
    }

    // ---- Blur compute ----
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = 3;
        layoutCI.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device, &layoutCI, nullptr,
                                        &blurSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create blur compute set layout!");

        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 1;
        plCI.pSetLayouts    = &blurSetLayout;
        if (vkCreatePipelineLayout(ctx.device, &plCI, nullptr,
                                   &blurPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create blur compute pipeline layout!");

        auto code = Pipeline::readFile("shaders/ssao_blur_comp.spv");
        VkShaderModule module = Pipeline::createShaderModule(ctx, code);

        VkComputePipelineCreateInfo cpCI{};
        cpCI.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpCI.stage.module = module;
        cpCI.stage.pName  = "main";
        cpCI.layout       = blurPipelineLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpCI,
                                     nullptr, &blurPipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create blur compute pipeline!");

        vkDestroyShaderModule(ctx.device, module, nullptr);

        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1}
        };
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = 2;
        poolCI.pPoolSizes    = poolSizes;
        if (vkCreateDescriptorPool(ctx.device, &poolCI, nullptr,
                                   &blurPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create blur descriptor pool!");
    }
}

// ================================================================
//  Descriptor set allocation
// ================================================================

void SSAO::allocateDescriptorSets(VulkanContext& ctx) {
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = ssaoPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &ssaoSetLayout;
        if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &ssaoSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate SSAO descriptor set!");
    }
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = blurPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &blurSetLayout;
        if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &blurSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate blur descriptor set!");
    }
}

// ================================================================
//  Size-dependent resources (recreated on resize)
// ================================================================

void SSAO::createSizeDependentResources(VulkanContext& ctx) {
    // Depth texture
    Image::createImage(ctx, width_, height_,
                       depthFormat_, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       depthImage, depthMem);
    depthView = Image::createImageView(ctx, depthImage, depthFormat_,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);

    // View-space normals texture
    Image::createImage(ctx, width_, height_,
                       VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       normalImage, normalMem);
    normalView = Image::createImageView(ctx, normalImage,
                                        VK_FORMAT_R16G16B16A16_SFLOAT,
                                        VK_IMAGE_ASPECT_COLOR_BIT);

    // Raw AO (compute storage + sampled by blur)
    Image::createImage(ctx, width_, height_,
                       VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       rawAOImage, rawAOMem);
    rawAOView = Image::createImageView(ctx, rawAOImage, VK_FORMAT_R8_UNORM,
                                       VK_IMAGE_ASPECT_COLOR_BIT);

    // Blurred AO (compute storage + sampled by main pass)
    Image::createImage(ctx, width_, height_,
                       VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       blurredAOImage, blurredAOMem);
    blurredAOView = Image::createImageView(ctx, blurredAOImage,
                                           VK_FORMAT_R8_UNORM,
                                           VK_IMAGE_ASPECT_COLOR_BIT);

    // Initial layout transitions
    Image::transitionImageLayout(ctx, rawAOImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL, 1, 1);
    Image::transitionImageLayout(ctx, blurredAOImage,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);

    // Pre-pass framebuffer
    std::array<VkImageView, 2> attachments = { normalView, depthView };
    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass      = prePassRenderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbInfo.pAttachments    = attachments.data();
    fbInfo.width           = width_;
    fbInfo.height          = height_;
    fbInfo.layers          = 1;
    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr,
                            &prePassFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create SSAO pre-pass framebuffer!");
}

void SSAO::destroySizeDependentResources(VulkanContext& ctx) {
    vkDestroyFramebuffer(ctx.device, prePassFramebuffer, nullptr);

    vkDestroyImageView(ctx.device, blurredAOView, nullptr);
    vmaDestroyImage(ctx.allocator, blurredAOImage, blurredAOMem);

    vkDestroyImageView(ctx.device, rawAOView, nullptr);
    vmaDestroyImage(ctx.allocator, rawAOImage, rawAOMem);

    vkDestroyImageView(ctx.device, normalView, nullptr);
    vmaDestroyImage(ctx.allocator, normalImage, normalMem);

    vkDestroyImageView(ctx.device, depthView, nullptr);
    vmaDestroyImage(ctx.allocator, depthImage, depthMem);
}

// ================================================================
//  Descriptor set writes (called on create + resize)
// ================================================================

void SSAO::writeDescriptorSets(VulkanContext& ctx) {
    // ---- SSAO compute set ----
    VkDescriptorImageInfo depthInfo{};
    depthInfo.sampler     = nearestSampler;
    depthInfo.imageView   = depthView;
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalInfo{};
    normalInfo.sampler     = nearestSampler;
    normalInfo.imageView   = normalView;
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo noiseInfo{};
    noiseInfo.sampler     = noiseSampler;
    noiseInfo.imageView   = noiseView;
    noiseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo kernelInfo{};
    kernelInfo.buffer = kernelBuffer;
    kernelInfo.offset = 0;
    kernelInfo.range  = sizeof(glm::vec4) * KERNEL_SIZE;

    VkDescriptorBufferInfo paramsInfo{};
    paramsInfo.buffer = paramsBuffer;
    paramsInfo.offset = 0;
    paramsInfo.range  = sizeof(SSAOParams);

    VkDescriptorImageInfo rawAOStorageInfo{};
    rawAOStorageInfo.imageView   = rawAOView;
    rawAOStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> ssaoWrites{};
    ssaoWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[0].dstSet          = ssaoSet;
    ssaoWrites[0].dstBinding      = 0;
    ssaoWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssaoWrites[0].descriptorCount = 1;
    ssaoWrites[0].pImageInfo      = &depthInfo;

    ssaoWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[1].dstSet          = ssaoSet;
    ssaoWrites[1].dstBinding      = 1;
    ssaoWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssaoWrites[1].descriptorCount = 1;
    ssaoWrites[1].pImageInfo      = &normalInfo;

    ssaoWrites[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[2].dstSet          = ssaoSet;
    ssaoWrites[2].dstBinding      = 2;
    ssaoWrites[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssaoWrites[2].descriptorCount = 1;
    ssaoWrites[2].pImageInfo      = &noiseInfo;

    ssaoWrites[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[3].dstSet          = ssaoSet;
    ssaoWrites[3].dstBinding      = 3;
    ssaoWrites[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ssaoWrites[3].descriptorCount = 1;
    ssaoWrites[3].pBufferInfo     = &kernelInfo;

    ssaoWrites[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[4].dstSet          = ssaoSet;
    ssaoWrites[4].dstBinding      = 4;
    ssaoWrites[4].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ssaoWrites[4].descriptorCount = 1;
    ssaoWrites[4].pBufferInfo     = &paramsInfo;

    ssaoWrites[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssaoWrites[5].dstSet          = ssaoSet;
    ssaoWrites[5].dstBinding      = 5;
    ssaoWrites[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ssaoWrites[5].descriptorCount = 1;
    ssaoWrites[5].pImageInfo      = &rawAOStorageInfo;

    vkUpdateDescriptorSets(ctx.device,
                           static_cast<uint32_t>(ssaoWrites.size()),
                           ssaoWrites.data(), 0, nullptr);

    // ---- Blur compute set ----
    VkDescriptorImageInfo rawAOSamplerInfo{};
    rawAOSamplerInfo.sampler     = nearestSampler;
    rawAOSamplerInfo.imageView   = rawAOView;
    rawAOSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo blurDepthInfo{};
    blurDepthInfo.sampler     = nearestSampler;
    blurDepthInfo.imageView   = depthView;
    blurDepthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo blurredAOStorageInfo{};
    blurredAOStorageInfo.imageView   = blurredAOView;
    blurredAOStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> blurWrites{};
    blurWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    blurWrites[0].dstSet          = blurSet;
    blurWrites[0].dstBinding      = 0;
    blurWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    blurWrites[0].descriptorCount = 1;
    blurWrites[0].pImageInfo      = &rawAOSamplerInfo;

    blurWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    blurWrites[1].dstSet          = blurSet;
    blurWrites[1].dstBinding      = 1;
    blurWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    blurWrites[1].descriptorCount = 1;
    blurWrites[1].pImageInfo      = &blurDepthInfo;

    blurWrites[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    blurWrites[2].dstSet          = blurSet;
    blurWrites[2].dstBinding      = 2;
    blurWrites[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    blurWrites[2].descriptorCount = 1;
    blurWrites[2].pImageInfo      = &blurredAOStorageInfo;

    vkUpdateDescriptorSets(ctx.device,
                           static_cast<uint32_t>(blurWrites.size()),
                           blurWrites.data(), 0, nullptr);
}
