#include "Application.h"
#include "rhi/VulkanTypes.h"
#include "renderer/Renderer.h"

#include <chrono>

void Application::run() {
    init();
    mainLoop();
    cleanup();
}

void Application::init() {
    ctx.init(window.getHandle());

    swapChain.create(ctx, window.getHandle());
    pipeline.create(ctx, swapChain);
    swapChain.createFramebuffers(ctx, pipeline.renderPass);
    shadowMap.create(ctx, pipeline.frameSetLayout);

    textureMgr.loadTexture(ctx, TEXTURE_PATH);
    textureMgr2.loadTexture(ctx, TEXTURE_PATH_2);
    groundTexture.createSolidColor(ctx, 255, 255, 255);

    mesh.loadMesh(ctx, MODEL_PATH);
    mesh2.loadMesh(ctx, MODEL_PATH_2);
    mesh3.loadMesh(ctx, MODEL_PATH_2);
    groundPlane.createPlane(ctx, 10.0f);

    renderer.init(ctx, swapChain, pipeline, shadowMap, { &textureMgr, &textureMgr2, &groundTexture });
}

void Application::mainLoop() {
    auto startTime = std::chrono::high_resolution_clock::now();

    while (!window.shouldClose()) {
        glfwPollEvents();

        auto  now  = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        // 第一个模型：绕 Z 轴旋转，使用纹理 0
        glm::mat4 model1 = glm::rotate(glm::mat4(1.0f),
                                       time * glm::radians(90.0f),
                                       glm::vec3(0.0f, 0.0f, 1.0f));

        // 第二个模型：向右偏移 2 个单位，反向旋转，使用纹理 1
        glm::mat4 model2 = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f))
                         * glm::rotate(glm::mat4(1.0f),
                                       -time * glm::radians(90.0f),
                                       glm::vec3(0.0f, 0.0f, 1.0f));

        glm::mat4 model3 = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f))
                         * glm::rotate(glm::mat4(1.0f),
                                       -time * glm::radians(90.0f),
                                       glm::vec3(0.0f, 0.0f, 1.0f));

        glm::mat4 groundModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.5f));

        std::vector<RenderObject> objects = {
            { &groundPlane, 2, groundModel },
            { &mesh,  0, model1 },
            { &mesh2, 1, model2 },
            { &mesh3, 1, model3 },
        };

        renderer.drawFrame(ctx, window, swapChain, pipeline, shadowMap, objects);
    }
    vkDeviceWaitIdle(ctx.device);
}

void Application::cleanup() {
    renderer.destroy(ctx);
    mesh.destroy(ctx);
    mesh2.destroy(ctx);
    mesh3.destroy(ctx);
    groundPlane.destroy(ctx);
    textureMgr.destroy(ctx);
    textureMgr2.destroy(ctx);
    groundTexture.destroy(ctx);
    shadowMap.destroy(ctx);
    swapChain.destroy(ctx);
    pipeline.destroy(ctx);
    ctx.destroy();
}
