#include "renderer/Renderer.h"
#include "rhi/VulkanContext.h"
#include "rhi/SwapChain.h"
#include "renderer/Pipeline.h"
#include "renderer/ShadowMap.h"
#include "renderer/IBLResources.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"
#include "core/Window.h"

#include <stdexcept>
#include <array>
#include <algorithm>

void Renderer::init(VulkanContext& ctx, SwapChain& swapChain,
                    Pipeline& pipeline, ShadowMap& shadowMap,
                    IBLResources& ibl,
                    const std::vector<TextureManager*>& textures)
{
    createUniformBuffers(ctx);
    createDescriptorPool(ctx, static_cast<uint32_t>(textures.size()));
    createFrameDescriptorSets(ctx, pipeline, shadowMap, ibl);
    createMaterialDescriptorSets(ctx, pipeline, textures);
    createCommandBuffers(ctx);
    uint32_t sci = static_cast<uint32_t>(swapChain.images.size());
    createSyncObjects(ctx, std::max(1u, sci));
}

void Renderer::destroy(VulkanContext& ctx) {
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
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(swapChain, currentFrame, cameraPos, cameraTarget, farPlane);

    vkResetFences(ctx.device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex,
                        swapChain, pipeline, shadowMap, objects);

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
    uint32_t totalSets    = frameSets + materialSets;

    // shadow(1) + irradiance(1) + prefilter(1) + brdfLut(1) = 4 samplers per frame set
    // + 1 sampler per material set
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         frameSets };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * frameSets + materialSets };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets       = totalSets;

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void Renderer::createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                         ShadowMap& shadowMap, IBLResources& ibl)
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

        std::array<VkWriteDescriptorSet, 5> writes{};
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

        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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
                                   float farPlane) {
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

        vkCmdEndRenderPass(commandBuffer);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}
