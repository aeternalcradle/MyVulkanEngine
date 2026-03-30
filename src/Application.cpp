#include "Application.h"
#include "rhi/VulkanTypes.h"

#include <chrono>

Application::Application(std::unique_ptr<Scene> s, RenderMode mode)
    : startupMode(mode), scene(std::move(s)) {}

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
    iblResources.create(ctx, "assets/skybox/venice_sunset_4k.exr");
    ssao.create(ctx, pipeline, swapChain.extent.width, swapChain.extent.height);

    scene->init(ctx);

    renderer.init(ctx, swapChain, pipeline, shadowMap, iblResources, ssao,
                  scene->getTextures());
    renderer.setRenderMode(startupMode);
}

void Application::mainLoop() {
    auto startTime = std::chrono::high_resolution_clock::now();
    SceneSetup setup = scene->getSetup();

    while (!window.shouldClose()) {
        glfwPollEvents();

        auto  now  = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        auto objects = scene->update(time);

        renderer.drawFrame(ctx, window, swapChain, pipeline, shadowMap, ssao,
                           objects, setup.cameraPos, setup.cameraTarget,
                           setup.farPlane);
    }
    vkDeviceWaitIdle(ctx.device);
}

void Application::cleanup() {
    renderer.destroy(ctx);
    ssao.destroy(ctx);
    scene->cleanup(ctx);
    iblResources.destroy(ctx);
    shadowMap.destroy(ctx);
    swapChain.destroy(ctx);
    pipeline.destroy(ctx);
    ctx.destroy();
}
