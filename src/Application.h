#pragma once

#include "core/Window.h"
#include "rhi/VulkanContext.h"
#include "rhi/SwapChain.h"
#include "renderer/Pipeline.h"
#include "renderer/Renderer.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"

// 顶层应用类，负责按正确顺序初始化和销毁所有子系统
class Application {
public:
    void run();

private:
    Window         window;
    VulkanContext  ctx;
    SwapChain      swapChain;
    Pipeline       pipeline;
    Renderer       renderer;

    TextureManager textureMgr;
    TextureManager textureMgr2;
    MeshLoader     mesh;
    MeshLoader     mesh2;
    MeshLoader     mesh3;

    void init();
    void mainLoop();
    void cleanup();
};
