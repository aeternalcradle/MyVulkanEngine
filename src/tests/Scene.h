#pragma once

#include "rhi/VulkanTypes.h"
#include "renderer/Renderer.h"
#include <vector>

class VulkanContext;
class TextureManager;

struct SceneSetup {
    glm::vec3 cameraPos    = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f);
    float     farPlane     = 10.0f;
};

class Scene {
public:
    virtual ~Scene() = default;

    virtual void init(VulkanContext& ctx) = 0;
    virtual std::vector<TextureManager*> getTextures() = 0;
    virtual SceneSetup getSetup() const = 0;
    virtual std::vector<RenderObject> update(float time) = 0;
    virtual void cleanup(VulkanContext& ctx) = 0;
};
