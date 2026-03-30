#include "Application.h"
#include "tests/PBRSpheresScene.h"
#include "tests/BasicScene.h"

#include <iostream>
#include <cstdlib>
#include <memory>

int main() {
    // 在这里切换启动渲染方式：RenderMode::Forward 或 RenderMode::DeferredMVP
    RenderMode startupMode = RenderMode::Forward;

    Application app(std::make_unique<PBRSpheresScene>(), startupMode);
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
