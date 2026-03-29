#pragma once

#include "tests/Scene.h"
#include "resource/TextureManager.h"
#include "resource/MeshLoader.h"

class BasicScene : public Scene {
public:
    void init(VulkanContext& ctx) override;
    std::vector<TextureManager*> getTextures() override;
    SceneSetup getSetup() const override;
    std::vector<RenderObject> update(float time) override;
    void cleanup(VulkanContext& ctx) override;

private:
    TextureManager textureMgr;
    TextureManager textureMgr2;
    TextureManager groundTexture;
    MeshLoader     mesh;
    MeshLoader     mesh2;
    MeshLoader     mesh3;
    MeshLoader     groundPlane;
};
