#include "Application.h"
#include "rhi/VulkanTypes.h"

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

    textureMgr.loadTexture(ctx, TEXTURE_PATH);
    mesh.loadMesh(ctx, MODEL_PATH);

    renderer.init(ctx, swapChain, pipeline, textureMgr);
}

void Application::mainLoop() {
    while (!window.shouldClose()) {
        glfwPollEvents();
        renderer.drawFrame(ctx, window, swapChain, pipeline, mesh);
    }
    vkDeviceWaitIdle(ctx.device);
}

void Application::cleanup() {
    renderer.destroy(ctx);
    mesh.destroy(ctx);
    textureMgr.destroy(ctx);
    swapChain.destroy(ctx);
    pipeline.destroy(ctx);
    ctx.destroy();
}
