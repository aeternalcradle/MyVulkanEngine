#include "tests/PBRSpheresScene.h"
#include "rhi/VulkanContext.h"

void PBRSpheresScene::init(VulkanContext& ctx) {
    sphere.createSphere(ctx, 1.0f, 64, 32);
    groundPlane.createPlane(ctx, 20.0f);
    whiteTex.createSolidColor(ctx, 255, 255, 255);
}

std::vector<TextureManager*> PBRSpheresScene::getTextures() {
    return { &whiteTex };
}

SceneSetup PBRSpheresScene::getSetup() const {
    return {
        glm::vec3(0.0f, -10.0f, 5.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        50.0f
    };
}

std::vector<RenderObject> PBRSpheresScene::update(float /*time*/) {
    std::vector<RenderObject> objects;

    glm::mat4 ground = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -1.2f));
    objects.push_back({ &groundPlane, 0, ground, 0.0f, 0.9f });

    for (int row = 0; row < GRID; ++row) {
        for (int col = 0; col < GRID; ++col) {
            float x = (col - (GRID - 1) / 2.0f) * SPACING;
            float y = (row - (GRID - 1) / 2.0f) * SPACING;

            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, 0.0f));

            float metallic  = static_cast<float>(col) / (GRID - 1);
            float roughness = 0.1f + static_cast<float>(row) / (GRID - 1) * 0.8f;

            objects.push_back({ &sphere, 0, model, metallic, roughness });
        }
    }

    return objects;
}

void PBRSpheresScene::cleanup(VulkanContext& ctx) {
    sphere.destroy(ctx);
    groundPlane.destroy(ctx);
    whiteTex.destroy(ctx);
}
