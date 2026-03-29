#include "tests/BasicScene.h"
#include "rhi/VulkanContext.h"

void BasicScene::init(VulkanContext& ctx) {
    textureMgr.loadTexture(ctx, TEXTURE_PATH);
    textureMgr2.loadTexture(ctx, TEXTURE_PATH_2);
    groundTexture.createSolidColor(ctx, 255, 255, 255);

    mesh.loadMesh(ctx, MODEL_PATH);
    mesh2.loadMesh(ctx, MODEL_PATH_2);
    mesh3.loadMesh(ctx, MODEL_PATH_2);
    groundPlane.createPlane(ctx, 10.0f);
}

std::vector<TextureManager*> BasicScene::getTextures() {
    return { &textureMgr, &textureMgr2, &groundTexture };
}

SceneSetup BasicScene::getSetup() const {
    return { glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f), 10.0f };
}

std::vector<RenderObject> BasicScene::update(float time) {
    glm::mat4 model1 = glm::rotate(glm::mat4(1.0f),
                                    time * glm::radians(90.0f),
                                    glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 model2 = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f))
                      * glm::rotate(glm::mat4(1.0f),
                                    -time * glm::radians(90.0f),
                                    glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 model3 = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f))
                      * glm::rotate(glm::mat4(1.0f),
                                    -time * glm::radians(90.0f),
                                    glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 groundModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.5f));

    return {
        { &groundPlane, 2, groundModel, 0.0f, 0.8f },
        { &mesh,  0, model1,  0.0f, 0.5f },
        { &mesh2, 1, model2,  0.0f, 0.3f },
        { &mesh3, 1, model3,  0.0f, 0.6f },
    };
}

void BasicScene::cleanup(VulkanContext& ctx) {
    mesh.destroy(ctx);
    mesh2.destroy(ctx);
    mesh3.destroy(ctx);
    groundPlane.destroy(ctx);
    textureMgr.destroy(ctx);
    textureMgr2.destroy(ctx);
    groundTexture.destroy(ctx);
}
