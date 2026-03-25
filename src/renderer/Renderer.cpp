#include "renderer/Renderer.h"
#include "rhi/VulkanContext.h"
#include "rhi/SwapChain.h"
#include "renderer/Pipeline.h"
#include "renderer/ShadowMap.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"
#include "core/Window.h"

#include <stdexcept>
#include <array>

void Renderer::init(VulkanContext& ctx, SwapChain& /*swapChain*/,
                    Pipeline& pipeline, ShadowMap& shadowMap,
                    const std::vector<TextureManager*>& textures)
{
    createUniformBuffers(ctx);
    createDescriptorPool(ctx, static_cast<uint32_t>(textures.size()));
    createFrameDescriptorSets(ctx, pipeline, shadowMap);
    createMaterialDescriptorSets(ctx, pipeline, textures);
    createCommandBuffers(ctx);
    createSyncObjects(ctx);
}

void Renderer::destroy(VulkanContext& ctx) {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(ctx.device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(ctx.device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(ctx.device, inFlightFences[i], nullptr);
        vkDestroyBuffer(ctx.device, uniformBuffers[i], nullptr);
        vkFreeMemory(ctx.device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
}

void Renderer::drawFrame(VulkanContext& ctx, Window& window,
                         SwapChain& swapChain, Pipeline& pipeline,
                         ShadowMap& shadowMap,
                         const std::vector<RenderObject>& objects)
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

    updateUniformBuffer(swapChain, currentFrame);

    vkResetFences(ctx.device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex,
                        swapChain, pipeline, shadowMap, objects);

    VkSemaphore          waitSemaphores[]   = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[]       = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

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
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        ctx.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersMemory[i]);
        vkMapMemory(ctx.device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
    }
}

void Renderer::createDescriptorPool(VulkanContext& ctx, uint32_t numTextures) {
    uint32_t frameSets    = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    uint32_t materialSets = numTextures * frameSets;
    uint32_t totalSets    = frameSets + materialSets;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         frameSets };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frameSets + materialSets };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets       = totalSets;

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void Renderer::createFrameDescriptorSets(VulkanContext& ctx, Pipeline& pipeline,
                                         ShadowMap& shadowMap)
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

        std::array<VkWriteDescriptorSet, 2> writes{};
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

void Renderer::createSyncObjects(VulkanContext& ctx) {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(ctx.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create synchronization objects for a frame!");
    }
}

// ---- 每帧更新 ----

void Renderer::updateUniformBuffer(SwapChain& swapChain, uint32_t currentImage) {
    if (swapChain.extent.width == 0 || swapChain.extent.height == 0) return;

    UniformBufferObject ubo{};

    ubo.lightDir   = glm::vec3(1.0f, -1.0f, -1.0f);
    ubo.ambient    = 0.15f;
    ubo.lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    ubo.lightSize  = 0.5f;

    // 相机
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                           glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                swapChain.extent.width / (float)swapChain.extent.height,
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    // 光空间矩阵（方向光 → 正交投影）
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

            // Set 1: per-material (diffuse texture)
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.pipelineLayout, 1, 1,
                                    &materialDescriptorSets[obj.textureIndex][currentFrame],
                                    0, nullptr);

            PushConstants pc{ obj.transform };
            vkCmdPushConstants(commandBuffer, pipeline.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pc);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(obj.mesh->indices.size()), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}
