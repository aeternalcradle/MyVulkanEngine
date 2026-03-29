#pragma once

#include "tests/Scene.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"

class PBRSpheresScene : public Scene {
public:
    void init(VulkanContext& ctx) override;
    std::vector<TextureManager*> getTextures() override;
    SceneSetup getSetup() const override;
    std::vector<RenderObject> update(float time) override;
    void cleanup(VulkanContext& ctx) override;

private:
    static constexpr int GRID = 3;
    static constexpr float SPACING = 2.5f;

    MeshLoader     sphere;
    MeshLoader     groundPlane;
    TextureManager whiteTex;
};
