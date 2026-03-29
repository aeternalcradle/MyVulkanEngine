#include "Application.h"
#include "tests/PBRSpheresScene.h"
#include "tests/BasicScene.h"

#include <iostream>
#include <cstdlib>
#include <memory>

int main() {
    Application app(std::make_unique<PBRSpheresScene>());
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
