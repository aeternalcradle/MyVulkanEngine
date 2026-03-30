#include "renderer/Renderer.h"
#include "rhi/VulkanContext.h"
#include "rhi/SwapChain.h"
#include "renderer/Pipeline.h"
#include "renderer/ShadowMap.h"
#include "renderer/IBLResources.h"
#include "renderer/SSAO.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"
#include "core/Window.h"
#include "rhi/Image.h"

#include <stdexcept>
#include <array>
#include <algorithm>

void Renderer::init(VulkanContext& ctx, SwapChain& swapChain,
                    Pipeline& pipeline, ShadowMap& shadowMap,
                    IBLResources& ibl, SSAO& ssao,
                    const std::vector<TextureManager*>& textures)
{
    createUniformBuffers(ctx);
    createDescriptorPool(ctx, static_cast<uint32_t>(textures.size()));
    createFrameDescriptorSets(ctx, pipeline, shadowMap, ibl, ssao);
    createMaterialDescriptorSets(ctx, pipeline, textures);
    createDeferredSampler(ctx);
    createDeferredResources(ctx, swapChain, pipeline);
    createCommandBuffers(ctx);
    uint32_t sci = static_cast<uint32_t>(swapChain.images.size());
    createSyncObjects(ctx, std::max(1u, sci));
}

void Renderer::destroy(VulkanContext& ctx) {
    destroyDeferredResources(ctx);
    deferredDescriptorSets.clear();
    if (deferredSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, deferredSetLayout, nullptr);
        deferredSetLayout = VK_NULL_HANDLE;
    }
    if (deferredSampler != VK_NULL_HANDLE) {
        vkDestroySampler(ctx.device, deferredSampler, nullptr);
        deferredSampler = VK_NULL_HANDLE;
    }

    for (VkSemaphore s : renderFinishedSemaphores)
        vkDestroySemaphore(ctx.device, s, nullptr);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(ctx.device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(ctx.device, inFlightFences[i], nullptr);
        vmaUnmapMemory(ctx.allocator, uniformBuffersAlloc[i]);
        vmaDestroyBuffer(ctx.allocator, uniformBuffers[i], uniformBuffersAlloc[i]);
    }
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
}

void Renderer::drawFrame(VulkanContext& ctx, Window& window,
                         SwapChain& swapChain, Pipeline& pipeline,
                         ShadowMap& shadowMap,
                         SSAO& ssao,
                         const std::vector<RenderObject>& objects,
                         const glm::vec3& cameraPos,
                         const glm::vec3& cameraTarget,
                         float farPlane)
{
    vkWaitForFences(ctx.device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(ctx.device, swapChain.swapChain, UINT64_MAX,
                                            imageAvailableSemaphores[currentFrame],
                                            VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        swapChain.cleanup(ctx);
        swapChain.create(ctx, window.getHandle());
        swapChain.createFramebuffers(ctx, pipeline.renderPass);
        ssao.resize(ctx, swapChain.extent.width, swapChain.extent.height);
        createDeferredResources(ctx, swapChain, pipeline);
        updateAODescriptorSets(ctx, ssao);
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(swapChain, currentFrame, cameraPos, cameraTarget, farPlane, ssao);

    vkResetFences(ctx.device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex,
                        swapChain, pipeline, shadowMap, ssao, objects);

    VkSemaphore          waitSemaphores[]   = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[]       = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &commandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    if (vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("failed to submit draw command buffer!");

    VkSwapchainKHR swapChains[] = { swapChain.swapChain };
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapChains;
    presentInfo.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(ctx.presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.wasResized()) {
        window.resetResized();
        swapChain.cleanup(ctx);
        swapChain.create(ctx, window.getHandle());
        swapChain.createFramebuffers(ctx, pipeline.renderPass);
        ssao.resize(ctx, swapChain.extent.width, swapChain.extent.height);
        createDeferredResources(ctx, swapChain, pipeline);
        updateAODescriptorSets(ctx, ssao);
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---- 资源创建 ----

void Renderer::createUniformBuffers(VulkanContext& ctx) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersAlloc.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        ctx.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersAlloc[i]);
        vmaMapMemory(ctx.allocator, uniformBuffersAlloc[i], &uniformBuffersMapped[i]);
    }
}

void Renderer::createDescriptorPool(VulkanContext& ctx, uint32_t numTextures) {
    uint32_t frameSets    = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    uint32_t materialSets = numTextures * frameSets;
    uint32_t deferredSets = frameSets;
    uint32_t totalSets    = frameSets + materialSets + deferredSets;

    // shadow(1) + irradiance(1) + prefilter(1) + brdfLut(1) + ssao(1) + envMap(1) = 6 samplers per frame set
    // + 1 sampler per material set
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2 * frameSets };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 * frameSets + materialSets + 3 * frameSets };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets       = totalSets;

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void Renderer::createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                         ShadowMap& shadowMap, IBLResources& ibl,
                                         SSAO& ssao)
{
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, pipeline.frameSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts        = layouts.data();

    frameDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, frameDescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate frame descriptor sets!");

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(UniformBufferObject);

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        shadowInfo.imageView   = shadowMap.depthImageView;
        shadowInfo.sampler     = shadowMap.sampler;

        VkDescriptorImageInfo irradianceInfo{};
        irradianceInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        irradianceInfo.imageView   = ibl.irradianceView;
        irradianceInfo.sampler     = ibl.cubemapSampler;

        VkDescriptorImageInfo prefilterInfo{};
        prefilterInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        prefilterInfo.imageView   = ibl.prefilteredView;
        prefilterInfo.sampler     = ibl.cubemapSampler;

        VkDescriptorImageInfo brdfLutInfo{};
        brdfLutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        brdfLutInfo.imageView   = ibl.brdfLutView;
        brdfLutInfo.sampler     = ibl.brdfLutSampler;

        VkDescriptorImageInfo ssaoInfo{};
        ssaoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ssaoInfo.imageView   = ssao.blurredAOView;
        ssaoInfo.sampler     = ssao.aoSampler;

        VkDescriptorImageInfo envMapInfo{};
        envMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        envMapInfo.imageView   = ibl.envCubeView;
        envMapInfo.sampler     = ibl.cubemapSampler;

        std::array<VkWriteDescriptorSet, 7> writes{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = frameDescriptorSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &bufferInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = frameDescriptorSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo      = &shadowInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = frameDescriptorSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo      = &irradianceInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = frameDescriptorSets[i];
        writes[3].dstBinding      = 3;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo      = &prefilterInfo;

        writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet          = frameDescriptorSets[i];
        writes[4].dstBinding      = 4;
        writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].descriptorCount = 1;
        writes[4].pImageInfo      = &brdfLutInfo;

        writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet          = frameDescriptorSets[i];
        writes[5].dstBinding      = 5;
        writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].descriptorCount = 1;
        writes[5].pImageInfo      = &ssaoInfo;

        writes[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet          = frameDescriptorSets[i];
        writes[6].dstBinding      = 6;
        writes[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].descriptorCount = 1;
        writes[6].pImageInfo      = &envMapInfo;

        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Renderer::updateAODescriptorSets(VulkanContext& ctx, SSAO& ssao) {
    for (size_t i = 0; i < frameDescriptorSets.size(); ++i) {
        VkDescriptorImageInfo ssaoInfo{};
        ssaoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ssaoInfo.imageView   = ssao.blurredAOView;
        ssaoInfo.sampler     = ssao.aoSampler;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = frameDescriptorSets[i];
        write.dstBinding      = 5;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &ssaoInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }
}

void Renderer::createMaterialDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                            const std::vector<TextureManager*>& textures)
{
    uint32_t numTextures = static_cast<uint32_t>(textures.size());
    materialDescriptorSets.resize(numTextures);

    for (uint32_t t = 0; t < numTextures; t++) {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, pipeline.materialSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts        = layouts.data();

        materialDescriptorSets[t].resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(ctx.device, &allocInfo, materialDescriptorSets[t].data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate material descriptor sets!");

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView   = textures[t]->textureImageView;
            imageInfo.sampler     = textures[t]->textureSampler;

            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = materialDescriptorSets[t][i];
            write.dstBinding      = 0;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo      = &imageInfo;

            vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
        }
    }
}

void Renderer::createCommandBuffers(VulkanContext& ctx) {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = ctx.commandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(ctx.device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers!");
}

void Renderer::createSyncObjects(VulkanContext& ctx, uint32_t swapChainImageCount) {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(swapChainImageCount);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(ctx.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create synchronization objects for a frame!");
    }
    for (uint32_t i = 0; i < swapChainImageCount; i++) {
        if (vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create render-finished semaphore!");
    }
}

// ---- 每帧更新 ----

void Renderer::updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage,
                                   const glm::vec3& cameraPos,
                                   const glm::vec3& cameraTarget,
                                   float farPlane,
                                   SSAO& ssao) {
    if (swapChain.extent.width == 0 || swapChain.extent.height == 0) return;

    UniformBufferObject ubo{};

    ubo.lightDir   = glm::vec3(1.0f, -1.0f, -1.0f);
    ubo.ambient    = 0.15f;
    ubo.lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    ubo.lightSize  = 0.5f;

    ubo.camPos = cameraPos;

    ubo.view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                swapChain.extent.width / (float)swapChain.extent.height,
                                0.1f, farPlane);
    ubo.proj[1][1] *= -1;
    ssao.updateParams(ubo.proj, glm::vec2(
        static_cast<float>(swapChain.extent.width),
        static_cast<float>(swapChain.extent.height)));

    glm::vec3 lightDirN = glm::normalize(ubo.lightDir);
    glm::vec3 lightPos  = -lightDirN * 6.0f;
    glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 lightProj = glm::ortho(-8.0f, 8.0f, -8.0f, 8.0f, 0.1f, 20.0f);
    lightProj[1][1] *= -1;
    ubo.lightViewProj = lightProj * lightView;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// ---- 命令录制（Shadow Pass + Main Pass）----

void Renderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                                   SwapChain& swapChain, Pipeline& pipeline,
                                   ShadowMap& shadowMap,
                                   SSAO& ssao,
                                   const std::vector<RenderObject>& objects)
{
    if (renderMode == RenderMode::DeferredMVP) {
        recordDeferredCommandBuffer(commandBuffer, imageIndex, swapChain, pipeline,
                                    shadowMap, ssao, objects);
        return;
    }

    recordForwardCommandBuffer(commandBuffer, imageIndex, swapChain, pipeline,
                               shadowMap, ssao, objects);
}

void Renderer::recordForwardCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                                          SwapChain& swapChain, Pipeline& pipeline,
                                          ShadowMap& shadowMap,
                                          SSAO& ssao,
                                          const std::vector<RenderObject>& objects)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

    // ======== Shadow Pass ========
    {
        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = shadowMap.renderPass;
        rpInfo.framebuffer       = shadowMap.framebuffer;
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = { ShadowMap::SHADOW_MAP_RESOLUTION, ShadowMap::SHADOW_MAP_RESOLUTION };
        rpInfo.clearValueCount   = 1;
        rpInfo.pClearValues      = &shadowClear;

        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMap.shadowPipeline);

        VkViewport viewport{};
        viewport.width    = (float)ShadowMap::SHADOW_MAP_RESOLUTION;
        viewport.height   = (float)ShadowMap::SHADOW_MAP_RESOLUTION;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, { ShadowMap::SHADOW_MAP_RESOLUTION, ShadowMap::SHADOW_MAP_RESOLUTION }};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                shadowMap.shadowPipelineLayout, 0, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);

        for (const auto& obj : objects) {
            VkBuffer     vertexBuffers[] = { obj.mesh->vertexBuffer };
            VkDeviceSize offsets[]       = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, obj.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            PushConstants pc{ obj.transform };
            vkCmdPushConstants(commandBuffer, shadowMap.shadowPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pc);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(obj.mesh->indices.size()), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);
    }

    // ======== Depth+Normal Pre-Pass + SSAO Compute ========
    ssao.recordPrePass(commandBuffer, frameDescriptorSets[currentFrame], objects);
    ssao.recordCompute(commandBuffer);

    // ======== Main Pass ========
    {
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = pipeline.renderPass;
        rpInfo.framebuffer       = swapChain.framebuffers[imageIndex];
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = swapChain.extent;
        rpInfo.clearValueCount   = static_cast<uint32_t>(clearValues.size());
        rpInfo.pClearValues      = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.graphicsPipeline);

        VkViewport viewport{};
        viewport.width    = (float)swapChain.extent.width;
        viewport.height   = (float)swapChain.extent.height;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapChain.extent};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Set 0: per-frame (UBO + shadow map)
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline.pipelineLayout, 0, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);

        for (const auto& obj : objects) {
            VkBuffer     vertexBuffers[] = { obj.mesh->vertexBuffer };
            VkDeviceSize offsets[]       = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, obj.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.pipelineLayout, 1, 1,
                                    &materialDescriptorSets[obj.textureIndex][currentFrame],
                                    0, nullptr);

            PushConstants pc{};
            pc.model     = obj.transform;
            pc.metallic  = obj.metallic;
            pc.roughness = obj.roughness;
            vkCmdPushConstants(commandBuffer, pipeline.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(obj.mesh->indices.size()), 1, 0, 0, 0);
        }

        // ---- Skybox (drawn last, depth test LEQUAL, no depth write) ----
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.skyboxPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline.skyboxPipelineLayout, 0, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffer, 36, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
    
}

void Renderer::createDeferredSampler(VulkanContext& ctx) {
    if (deferredSampler != VK_NULL_HANDLE) return;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter               = VK_FILTER_NEAREST;
    samplerInfo.minFilter               = VK_FILTER_NEAREST;
    samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable        = VK_FALSE;
    samplerInfo.maxAnisotropy           = 1.0f;
    samplerInfo.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable           = VK_FALSE;
    samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &deferredSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred sampler!");
}

void Renderer::destroyDeferredResources(VulkanContext& ctx) {
    for (VkFramebuffer fb : deferredLightingFramebuffers)
        vkDestroyFramebuffer(ctx.device, fb, nullptr);
    deferredLightingFramebuffers.clear();

    for (VkFramebuffer fb : deferredGeometryFramebuffers)
        vkDestroyFramebuffer(ctx.device, fb, nullptr);
    deferredGeometryFramebuffers.clear();

    if (deferredLightingPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, deferredLightingPipeline, nullptr);
        deferredLightingPipeline = VK_NULL_HANDLE;
    }
    if (deferredSkyboxPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, deferredSkyboxPipeline, nullptr);
        deferredSkyboxPipeline = VK_NULL_HANDLE;
    }
    if (deferredLightingPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, deferredLightingPipelineLayout, nullptr);
        deferredLightingPipelineLayout = VK_NULL_HANDLE;
    }
    if (deferredSkyboxPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, deferredSkyboxPipelineLayout, nullptr);
        deferredSkyboxPipelineLayout = VK_NULL_HANDLE;
    }
    if (deferredLightingRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx.device, deferredLightingRenderPass, nullptr);
        deferredLightingRenderPass = VK_NULL_HANDLE;
    }

    if (deferredGeometryPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, deferredGeometryPipeline, nullptr);
        deferredGeometryPipeline = VK_NULL_HANDLE;
    }
    if (deferredGeometryPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, deferredGeometryPipelineLayout, nullptr);
        deferredGeometryPipelineLayout = VK_NULL_HANDLE;
    }
    if (deferredGeometryRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx.device, deferredGeometryRenderPass, nullptr);
        deferredGeometryRenderPass = VK_NULL_HANDLE;
    }

    if (gbufferAlbedoView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, gbufferAlbedoView, nullptr);
        gbufferAlbedoView = VK_NULL_HANDLE;
    }
    if (gbufferNormalView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, gbufferNormalView, nullptr);
        gbufferNormalView = VK_NULL_HANDLE;
    }
    if (gbufferPositionView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, gbufferPositionView, nullptr);
        gbufferPositionView = VK_NULL_HANDLE;
    }
    if (deferredDepthView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, deferredDepthView, nullptr);
        deferredDepthView = VK_NULL_HANDLE;
    }

    if (gbufferAlbedoImage != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, gbufferAlbedoImage, gbufferAlbedoAlloc);
        gbufferAlbedoImage = VK_NULL_HANDLE;
        gbufferAlbedoAlloc = VK_NULL_HANDLE;
    }
    if (gbufferNormalImage != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, gbufferNormalImage, gbufferNormalAlloc);
        gbufferNormalImage = VK_NULL_HANDLE;
        gbufferNormalAlloc = VK_NULL_HANDLE;
    }
    if (gbufferPositionImage != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, gbufferPositionImage, gbufferPositionAlloc);
        gbufferPositionImage = VK_NULL_HANDLE;
        gbufferPositionAlloc = VK_NULL_HANDLE;
    }
    if (deferredDepthImage != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, deferredDepthImage, deferredDepthAlloc);
        deferredDepthImage = VK_NULL_HANDLE;
        deferredDepthAlloc = VK_NULL_HANDLE;
    }
}

void Renderer::createDeferredResources(VulkanContext& ctx, SwapChain& swapChain,
                                       Pipeline& pipeline) {
    destroyDeferredResources(ctx);

    if (deferredSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding         = 0;
        uboBinding.descriptorCount = 1;
        uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding albedoBinding{};
        albedoBinding.binding         = 1;
        albedoBinding.descriptorCount = 1;
        albedoBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        albedoBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding normalBinding{};
        normalBinding.binding         = 2;
        normalBinding.descriptorCount = 1;
        normalBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding positionBinding{};
        positionBinding.binding         = 3;
        positionBinding.descriptorCount = 1;
        positionBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        positionBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 4> bindings = {
            uboBinding, albedoBinding, normalBinding, positionBinding
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings    = bindings.data();

        if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &deferredSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create deferred descriptor set layout!");
    }

    Image::createImage(ctx, swapChain.extent.width, swapChain.extent.height,
                       VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       gbufferAlbedoImage, gbufferAlbedoAlloc);
    gbufferAlbedoView = Image::createImageView(ctx, gbufferAlbedoImage,
                                               VK_FORMAT_R8G8B8A8_UNORM,
                                               VK_IMAGE_ASPECT_COLOR_BIT);

    Image::createImage(ctx, swapChain.extent.width, swapChain.extent.height,
                       VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       gbufferNormalImage, gbufferNormalAlloc);
    gbufferNormalView = Image::createImageView(ctx, gbufferNormalImage,
                                               VK_FORMAT_R16G16B16A16_SFLOAT,
                                               VK_IMAGE_ASPECT_COLOR_BIT);

    Image::createImage(ctx, swapChain.extent.width, swapChain.extent.height,
                       VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       gbufferPositionImage, gbufferPositionAlloc);
    gbufferPositionView = Image::createImageView(ctx, gbufferPositionImage,
                                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                                 VK_IMAGE_ASPECT_COLOR_BIT);

    VkFormat depthFormat = ctx.findDepthFormat();
    Image::createImage(ctx, swapChain.extent.width, swapChain.extent.height,
                       depthFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       deferredDepthImage, deferredDepthAlloc);
    deferredDepthView = Image::createImageView(ctx, deferredDepthImage,
                                               depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkAttachmentDescription albedoAttachment{};
    albedoAttachment.format         = VK_FORMAT_R8G8B8A8_UNORM;
    albedoAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    albedoAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    albedoAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    albedoAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    albedoAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    albedoAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    albedoAttachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription normalAttachment = albedoAttachment;
    normalAttachment.format                  = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkAttachmentDescription positionAttachment = albedoAttachment;
    positionAttachment.format                  = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = depthFormat;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference albedoRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference normalRef{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference positionRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    std::array<VkAttachmentReference, 3> colorRefs = { albedoRef, normalRef, positionRef };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments       = colorRefs.data();
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 4> gbufferAttachments = {
        albedoAttachment, normalAttachment, positionAttachment, depthAttachment
    };

    VkRenderPassCreateInfo gbufferRpInfo{};
    gbufferRpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    gbufferRpInfo.attachmentCount = static_cast<uint32_t>(gbufferAttachments.size());
    gbufferRpInfo.pAttachments    = gbufferAttachments.data();
    gbufferRpInfo.subpassCount    = 1;
    gbufferRpInfo.pSubpasses      = &subpass;
    gbufferRpInfo.dependencyCount = 1;
    gbufferRpInfo.pDependencies   = &dep;

    if (vkCreateRenderPass(ctx.device, &gbufferRpInfo, nullptr, &deferredGeometryRenderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred geometry render pass!");

    auto gVertCode = Pipeline::readFile("shaders/deferred_gbuffer_vert.spv");
    auto gFragCode = Pipeline::readFile("shaders/deferred_gbuffer_frag.spv");
    VkShaderModule gVertModule = Pipeline::createShaderModule(ctx, gVertCode);
    VkShaderModule gFragModule = Pipeline::createShaderModule(ctx, gFragCode);

    VkPipelineShaderStageCreateInfo gVertStage{};
    gVertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gVertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    gVertStage.module = gVertModule;
    gVertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo gFragStage{};
    gFragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gFragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    gFragStage.module = gFragModule;
    gFragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo gStages[] = { gVertStage, gFragStage };

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

    std::array<VkPipelineColorBlendAttachmentState, 3> gbufferBlendAttachments{};
    for (auto& blend : gbufferBlendAttachments) {
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable    = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = static_cast<uint32_t>(gbufferBlendAttachments.size());
    colorBlending.pAttachments    = gbufferBlendAttachments.data();

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(PushConstants);

    VkDescriptorSetLayout gbufferLayouts[] = { pipeline.frameSetLayout, pipeline.materialSetLayout };

    VkPipelineLayoutCreateInfo gLayoutInfo{};
    gLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gLayoutInfo.setLayoutCount         = 2;
    gLayoutInfo.pSetLayouts            = gbufferLayouts;
    gLayoutInfo.pushConstantRangeCount = 1;
    gLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(ctx.device, &gLayoutInfo, nullptr, &deferredGeometryPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred geometry pipeline layout!");

    VkGraphicsPipelineCreateInfo gPipelineInfo{};
    gPipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gPipelineInfo.stageCount          = 2;
    gPipelineInfo.pStages             = gStages;
    gPipelineInfo.pVertexInputState   = &vertexInput;
    gPipelineInfo.pInputAssemblyState = &inputAssembly;
    gPipelineInfo.pViewportState      = &viewportState;
    gPipelineInfo.pRasterizationState = &rasterizer;
    gPipelineInfo.pMultisampleState   = &multisampling;
    gPipelineInfo.pDepthStencilState  = &depthStencil;
    gPipelineInfo.pColorBlendState    = &colorBlending;
    gPipelineInfo.pDynamicState       = &dynamicState;
    gPipelineInfo.layout              = deferredGeometryPipelineLayout;
    gPipelineInfo.renderPass          = deferredGeometryRenderPass;
    gPipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gPipelineInfo, nullptr, &deferredGeometryPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred geometry pipeline!");

    vkDestroyShaderModule(ctx.device, gFragModule, nullptr);
    vkDestroyShaderModule(ctx.device, gVertModule, nullptr);

    deferredGeometryFramebuffers.resize(swapChain.imageViews.size());
    for (size_t i = 0; i < deferredGeometryFramebuffers.size(); ++i) {
        std::array<VkImageView, 4> attachments = {
            gbufferAlbedoView,
            gbufferNormalView,
            gbufferPositionView,
            deferredDepthView,
        };

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = deferredGeometryRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments    = attachments.data();
        fbInfo.width           = swapChain.extent.width;
        fbInfo.height          = swapChain.extent.height;
        fbInfo.layers          = 1;

        if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &deferredGeometryFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create deferred geometry framebuffer!");
    }

    VkAttachmentDescription lightingColor{};
    lightingColor.format         = swapChain.imageFormat;
    lightingColor.samples        = VK_SAMPLE_COUNT_1_BIT;
    lightingColor.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    lightingColor.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    lightingColor.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    lightingColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    lightingColor.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    lightingColor.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference lightingColorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription lightingSubpass{};
    lightingSubpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    lightingSubpass.colorAttachmentCount = 1;
    lightingSubpass.pColorAttachments    = &lightingColorRef;

    VkSubpassDependency lightingDep{};
    lightingDep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    lightingDep.dstSubpass    = 0;
    lightingDep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    lightingDep.srcAccessMask = 0;
    lightingDep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    lightingDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo lightingRpInfo{};
    lightingRpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    lightingRpInfo.attachmentCount = 1;
    lightingRpInfo.pAttachments    = &lightingColor;
    lightingRpInfo.subpassCount    = 1;
    lightingRpInfo.pSubpasses      = &lightingSubpass;
    lightingRpInfo.dependencyCount = 1;
    lightingRpInfo.pDependencies   = &lightingDep;

    if (vkCreateRenderPass(ctx.device, &lightingRpInfo, nullptr, &deferredLightingRenderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred lighting render pass!");

    auto lVertCode = Pipeline::readFile("shaders/deferred_lighting_vert.spv");
    auto lFragCode = Pipeline::readFile("shaders/deferred_lighting_frag.spv");
    VkShaderModule lVertModule = Pipeline::createShaderModule(ctx, lVertCode);
    VkShaderModule lFragModule = Pipeline::createShaderModule(ctx, lFragCode);

    auto sVertCode = Pipeline::readFile("shaders/deferred_skybox_vert.spv");
    auto sFragCode = Pipeline::readFile("shaders/deferred_skybox_frag.spv");
    VkShaderModule sVertModule = Pipeline::createShaderModule(ctx, sVertCode);
    VkShaderModule sFragModule = Pipeline::createShaderModule(ctx, sFragCode);

    VkPipelineShaderStageCreateInfo lVertStage{};
    lVertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    lVertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    lVertStage.module = lVertModule;
    lVertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo lFragStage{};
    lFragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    lFragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    lFragStage.module = lFragModule;
    lFragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo lStages[] = { lVertStage, lFragStage };

    VkPipelineShaderStageCreateInfo sVertStage{};
    sVertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    sVertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    sVertStage.module = sVertModule;
    sVertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo sFragStage{};
    sFragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    sFragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    sFragStage.module = sFragModule;
    sFragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo sStages[] = { sVertStage, sFragStage };

    VkPipelineVertexInputStateCreateInfo fsVertexInput{};
    fsVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo fsInputAssembly{};
    fsInputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    fsInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo fsViewportState{};
    fsViewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    fsViewportState.viewportCount = 1;
    fsViewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo fsRasterizer{};
    fsRasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    fsRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    fsRasterizer.lineWidth   = 1.0f;
    fsRasterizer.cullMode    = VK_CULL_MODE_NONE;
    fsRasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo fsMultisampling{};
    fsMultisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    fsMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo fsDepthStencil{};
    fsDepthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    fsDepthStencil.depthTestEnable  = VK_FALSE;
    fsDepthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState fsBlendAttachment{};
    fsBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    fsBlendAttachment.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo fsColorBlend{};
    fsColorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    fsColorBlend.attachmentCount = 1;
    fsColorBlend.pAttachments    = &fsBlendAttachment;

    VkPipelineDynamicStateCreateInfo fsDynamicState{};
    fsDynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    fsDynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    fsDynamicState.pDynamicStates    = dynamicStates.data();

    VkDescriptorSetLayout lightingLayouts[] = { deferredSetLayout, pipeline.frameSetLayout };

    VkPipelineLayoutCreateInfo lLayoutInfo{};
    lLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lLayoutInfo.setLayoutCount = 2;
    lLayoutInfo.pSetLayouts    = lightingLayouts;

    if (vkCreatePipelineLayout(ctx.device, &lLayoutInfo, nullptr, &deferredLightingPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred lighting pipeline layout!");

    VkGraphicsPipelineCreateInfo lPipelineInfo{};
    lPipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    lPipelineInfo.stageCount          = 2;
    lPipelineInfo.pStages             = lStages;
    lPipelineInfo.pVertexInputState   = &fsVertexInput;
    lPipelineInfo.pInputAssemblyState = &fsInputAssembly;
    lPipelineInfo.pViewportState      = &fsViewportState;
    lPipelineInfo.pRasterizationState = &fsRasterizer;
    lPipelineInfo.pMultisampleState   = &fsMultisampling;
    lPipelineInfo.pDepthStencilState  = &fsDepthStencil;
    lPipelineInfo.pColorBlendState    = &fsColorBlend;
    lPipelineInfo.pDynamicState       = &fsDynamicState;
    lPipelineInfo.layout              = deferredLightingPipelineLayout;
    lPipelineInfo.renderPass          = deferredLightingRenderPass;
    lPipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &lPipelineInfo, nullptr, &deferredLightingPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred lighting pipeline!");

    VkPipelineColorBlendAttachmentState skyboxBlendAttachment{};
    skyboxBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    skyboxBlendAttachment.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo skyboxColorBlend{};
    skyboxColorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    skyboxColorBlend.attachmentCount = 1;
    skyboxColorBlend.pAttachments    = &skyboxBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo skyboxDepthStencil{};
    skyboxDepthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    skyboxDepthStencil.depthTestEnable  = VK_FALSE;
    skyboxDepthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineLayoutCreateInfo skyboxLayoutInfo{};
    skyboxLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    skyboxLayoutInfo.setLayoutCount = 2;
    skyboxLayoutInfo.pSetLayouts    = lightingLayouts;

    if (vkCreatePipelineLayout(ctx.device, &skyboxLayoutInfo, nullptr, &deferredSkyboxPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred skybox pipeline layout!");

    VkGraphicsPipelineCreateInfo sPipelineInfo{};
    sPipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    sPipelineInfo.stageCount          = 2;
    sPipelineInfo.pStages             = sStages;
    sPipelineInfo.pVertexInputState   = &fsVertexInput;
    sPipelineInfo.pInputAssemblyState = &fsInputAssembly;
    sPipelineInfo.pViewportState      = &fsViewportState;
    sPipelineInfo.pRasterizationState = &fsRasterizer;
    sPipelineInfo.pMultisampleState   = &fsMultisampling;
    sPipelineInfo.pDepthStencilState  = &skyboxDepthStencil;
    sPipelineInfo.pColorBlendState    = &skyboxColorBlend;
    sPipelineInfo.pDynamicState       = &fsDynamicState;
    sPipelineInfo.layout              = deferredSkyboxPipelineLayout;
    sPipelineInfo.renderPass          = deferredLightingRenderPass;
    sPipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &sPipelineInfo, nullptr, &deferredSkyboxPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create deferred skybox pipeline!");

    vkDestroyShaderModule(ctx.device, lFragModule, nullptr);
    vkDestroyShaderModule(ctx.device, lVertModule, nullptr);
    vkDestroyShaderModule(ctx.device, sFragModule, nullptr);
    vkDestroyShaderModule(ctx.device, sVertModule, nullptr);

    deferredLightingFramebuffers.resize(swapChain.imageViews.size());
    for (size_t i = 0; i < deferredLightingFramebuffers.size(); ++i) {
        VkImageView attachment = swapChain.imageViews[i];

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = deferredLightingRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments    = &attachment;
        fbInfo.width           = swapChain.extent.width;
        fbInfo.height          = swapChain.extent.height;
        fbInfo.layers          = 1;

        if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &deferredLightingFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create deferred lighting framebuffer!");
    }

    createDeferredDescriptorSets(ctx);
}

void Renderer::createDeferredDescriptorSets(VulkanContext& ctx) {
    if (deferredDescriptorSets.empty()) {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, deferredSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts        = layouts.data();

        deferredDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(ctx.device, &allocInfo, deferredDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate deferred descriptor sets!");
    }

    for (size_t i = 0; i < deferredDescriptorSets.size(); ++i) {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uniformBuffers[i];
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(UniformBufferObject);

        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        albedoInfo.imageView   = gbufferAlbedoView;
        albedoInfo.sampler     = deferredSampler;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalInfo.imageView   = gbufferNormalView;
        normalInfo.sampler     = deferredSampler;

        VkDescriptorImageInfo positionInfo{};
        positionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        positionInfo.imageView   = gbufferPositionView;
        positionInfo.sampler     = deferredSampler;

        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = deferredDescriptorSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &uboInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = deferredDescriptorSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo      = &albedoInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = deferredDescriptorSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo      = &normalInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = deferredDescriptorSets[i];
        writes[3].dstBinding      = 3;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo      = &positionInfo;

        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Renderer::recordDeferredCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex,
                                           SwapChain& swapChain, Pipeline& pipeline,
                                           ShadowMap& shadowMap, SSAO& ssao,
                                           const std::vector<RenderObject>& objects)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording deferred command buffer!");

    // ======== Shadow Pass ========
    {
        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = shadowMap.renderPass;
        rpInfo.framebuffer       = shadowMap.framebuffer;
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = { ShadowMap::SHADOW_MAP_RESOLUTION, ShadowMap::SHADOW_MAP_RESOLUTION };
        rpInfo.clearValueCount   = 1;
        rpInfo.pClearValues      = &shadowClear;

        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMap.shadowPipeline);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(ShadowMap::SHADOW_MAP_RESOLUTION);
        viewport.height   = static_cast<float>(ShadowMap::SHADOW_MAP_RESOLUTION);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, { ShadowMap::SHADOW_MAP_RESOLUTION, ShadowMap::SHADOW_MAP_RESOLUTION }};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                shadowMap.shadowPipelineLayout, 0, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);

        for (const auto& obj : objects) {
            VkBuffer     vertexBuffers[] = { obj.mesh->vertexBuffer };
            VkDeviceSize offsets[]       = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, obj.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            PushConstants pc{ obj.transform };
            vkCmdPushConstants(commandBuffer, shadowMap.shadowPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pc);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(obj.mesh->indices.size()), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);
    }

    // ======== SSAO Pre-pass + Compute ========
    ssao.recordPrePass(commandBuffer, frameDescriptorSets[currentFrame], objects);
    ssao.recordCompute(commandBuffer);

    {
        std::array<VkClearValue, 4> clearValues{};
        clearValues[0].color        = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].color        = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[2].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[3].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = deferredGeometryRenderPass;
        rpInfo.framebuffer       = deferredGeometryFramebuffers[imageIndex];
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = swapChain.extent;
        rpInfo.clearValueCount   = static_cast<uint32_t>(clearValues.size());
        rpInfo.pClearValues      = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredGeometryPipeline);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapChain.extent.width);
        viewport.height   = static_cast<float>(swapChain.extent.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapChain.extent};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredGeometryPipelineLayout, 0, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);

        for (const auto& obj : objects) {
            VkBuffer     vertexBuffers[] = { obj.mesh->vertexBuffer };
            VkDeviceSize offsets[]       = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, obj.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    deferredGeometryPipelineLayout, 1, 1,
                                    &materialDescriptorSets[obj.textureIndex][currentFrame],
                                    0, nullptr);

            PushConstants pc{};
            pc.model     = obj.transform;
            pc.metallic  = obj.metallic;
            pc.roughness = obj.roughness;
            vkCmdPushConstants(commandBuffer, deferredGeometryPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(obj.mesh->indices.size()), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);
    }

    std::array<VkImageMemoryBarrier, 3> barriers{};
    VkImage images[] = { gbufferAlbedoImage, gbufferNormalImage, gbufferPositionImage };
    for (size_t i = 0; i < barriers.size(); ++i) {
        barriers[i].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].oldLayout                       = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[i].newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[i].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].image                           = images[i];
        barriers[i].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[i].subresourceRange.baseMipLevel   = 0;
        barriers[i].subresourceRange.levelCount     = 1;
        barriers[i].subresourceRange.baseArrayLayer = 0;
        barriers[i].subresourceRange.layerCount     = 1;
        barriers[i].srcAccessMask                   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[i].dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
    }

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         static_cast<uint32_t>(barriers.size()), barriers.data());

    {
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = deferredLightingRenderPass;
        rpInfo.framebuffer       = deferredLightingFramebuffers[imageIndex];
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = swapChain.extent;
        rpInfo.clearValueCount   = 1;
        rpInfo.pClearValues      = &clear;

        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredLightingPipeline);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapChain.extent.width);
        viewport.height   = static_cast<float>(swapChain.extent.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapChain.extent};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredLightingPipelineLayout, 0, 1,
                                &deferredDescriptorSets[currentFrame], 0, nullptr);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    deferredLightingPipelineLayout, 1, 1,
                    &frameDescriptorSets[currentFrame], 0, nullptr);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredSkyboxPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredSkyboxPipelineLayout, 0, 1,
                                &deferredDescriptorSets[currentFrame], 0, nullptr);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredSkyboxPipelineLayout, 1, 1,
                                &frameDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
    }

    std::array<VkImageMemoryBarrier, 3> resetBarriers{};
    for (size_t i = 0; i < resetBarriers.size(); ++i) {
        resetBarriers[i].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        resetBarriers[i].oldLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        resetBarriers[i].newLayout                       = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        resetBarriers[i].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[i].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[i].image                           = images[i];
        resetBarriers[i].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        resetBarriers[i].subresourceRange.baseMipLevel   = 0;
        resetBarriers[i].subresourceRange.levelCount     = 1;
        resetBarriers[i].subresourceRange.baseArrayLayer = 0;
        resetBarriers[i].subresourceRange.layerCount     = 1;
        resetBarriers[i].srcAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
        resetBarriers[i].dstAccessMask                   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         static_cast<uint32_t>(resetBarriers.size()), resetBarriers.data());

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record deferred command buffer!");
}
