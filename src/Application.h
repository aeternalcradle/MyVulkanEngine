#pragma once

#include "core/Window.h"
#include "rhi/VulkanContext.h"
#include "rhi/SwapChain.h"
#include "renderer/Pipeline.h"
#include "renderer/Renderer.h"
#include "renderer/ShadowMap.h"
#include "renderer/IBLResources.h"
#include "renderer/SSAO.h"
#include "tests/Scene.h"

#include <memory>

class Application {
public:
    explicit Application(std::unique_ptr<Scene> scene);
    void run();

private:
    Window         window;
    VulkanContext  ctx;
    SwapChain      swapChain;
    Pipeline       pipeline;
    ShadowMap      shadowMap;
    IBLResources   iblResources;
    SSAO           ssao;
    Renderer       renderer;

    std::unique_ptr<Scene> scene;

    void init();
    void mainLoop();
    void cleanup();
};
