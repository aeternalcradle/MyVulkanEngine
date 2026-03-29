#include "resource/MeshLoader.h"
#include "rhi/VulkanContext.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <stdexcept>
#include <unordered_map>

void MeshLoader::loadMesh(VulkanContext& ctx, const std::string& modelPath) {
    loadModel(modelPath);
    createVertexBuffer(ctx);
    createIndexBuffer(ctx);
}

void MeshLoader::createPlane(VulkanContext& ctx, float size) {
    float half = size / 2.0f;
    float tile = size;

    // Vertex 顺序：pos, normal, color, texCoord（XY 平面，法线 +Z）
    vertices = {
        {{ -half, -half, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f }},
        {{  half, -half, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.5f, 0.5f }, { tile, 0.0f }},
        {{  half,  half, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.5f, 0.5f }, { tile, tile }},
        {{ -half,  half, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.5f, 0.5f }, { 0.0f, tile }},
    };

    indices = { 0, 1, 2, 2, 3, 0 };

    createVertexBuffer(ctx);
    createIndexBuffer(ctx);
}

void MeshLoader::createSphere(VulkanContext& ctx, float radius,
                              uint32_t sectors, uint32_t stacks) {
    vertices.clear();
    indices.clear();

    const float PI = 3.14159265358979323846f;

    for (uint32_t i = 0; i <= stacks; ++i) {
        float stackAngle = PI / 2.0f - static_cast<float>(i) * PI / stacks;
        float xy = radius * cosf(stackAngle);
        float z  = radius * sinf(stackAngle);

        for (uint32_t j = 0; j <= sectors; ++j) {
            float sectorAngle = static_cast<float>(j) * 2.0f * PI / sectors;

            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);

            Vertex v{};
            v.pos    = { x, y, z };
            v.normal = glm::normalize(glm::vec3(x, y, z));
            v.color  = { 1.0f, 1.0f, 1.0f };
            v.texCoord = { static_cast<float>(j) / sectors,
                           static_cast<float>(i) / stacks };
            vertices.push_back(v);
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < sectors; ++j) {
            uint32_t first  = i * (sectors + 1) + j;
            uint32_t second = first + sectors + 1;

            if (i != 0) {
                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);
            }
            if (i != stacks - 1) {
                indices.push_back(first + 1);
                indices.push_back(second);
                indices.push_back(second + 1);
            }
        }
    }

    createVertexBuffer(ctx);
    createIndexBuffer(ctx);
}

void MeshLoader::destroy(VulkanContext& ctx) {
    vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAlloc);
    vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAlloc);
}

void MeshLoader::computeSmoothNormals() {
    for (auto& v : vertices)
        v.normal = glm::vec3(0.0f);

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];
        

        glm::vec3 p0 = vertices[i0].pos;
        glm::vec3 p1 = vertices[i1].pos;
        glm::vec3 p2 = vertices[i2].pos;
        glm::vec3 e1 = p1 - p0;
        glm::vec3 e2 = p2 - p0;
        glm::vec3 n  = glm::cross(e1, e2);
        float     len = glm::length(n);
        if (len > 1e-8f) {
            n /= len;
            vertices[i0].normal += n;
            vertices[i1].normal += n;
            vertices[i2].normal += n;
        }
    }

    for (auto& v : vertices) {
        float len = glm::length(v.normal);
        if (len > 1e-8f)
            v.normal /= len;
        else
            v.normal = glm::vec3(0.0f, 0.0f, 1.0f);
    }
}

void MeshLoader::loadModel(const std::string& modelPath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str()))
        throw std::runtime_error(warn + err);

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            if (index.vertex_index < 0)
                continue;

            const size_t vBase = static_cast<size_t>(index.vertex_index) * 3u;
            if (vBase + 2 >= attrib.vertices.size())
                throw std::runtime_error("OBJ vertex index out of range");

            Vertex vertex{};
            vertex.pos = {
                attrib.vertices[vBase + 0],
                attrib.vertices[vBase + 1],
                attrib.vertices[vBase + 2]
            };

            if (index.texcoord_index >= 0) {
                const size_t tBase = static_cast<size_t>(index.texcoord_index) * 2u;
                if (tBase + 1 < attrib.texcoords.size()) {
                    vertex.texCoord = {
                        attrib.texcoords[tBase + 0],
                        1.0f - attrib.texcoords[tBase + 1]
                    };
                } else {
                    vertex.texCoord = { 0.0f, 0.0f };
                }
            } else {
                vertex.texCoord = { 0.0f, 0.0f };
            }

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (!attrib.normals.empty() && index.normal_index >= 0) {
                const size_t nBase = static_cast<size_t>(index.normal_index) * 3u;
                if (nBase + 2 < attrib.normals.size()) {
                    vertex.normal = {
                        attrib.normals[nBase + 0],
                        attrib.normals[nBase + 1],
                        attrib.normals[nBase + 2]
                    };
                } else {
                    vertex.normal = { 0.0f, 0.0f, 0.0f };
                }
            } else {
                vertex.normal = { 0.0f, 0.0f, 0.0f };
            }

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }
            indices.push_back(uniqueVertices[vertex]);
        }
    }

    if (attrib.normals.empty()) {
        computeSmoothNormals();
    } else {
        bool needSmooth = false;
        for (const auto& v : vertices) {
            if (glm::dot(v.normal, v.normal) < 1e-12f) {
                needSmooth = true;
                break;
            }
        }
        if (needSmooth)
            computeSmoothNormals();
    }
}

void MeshLoader::createVertexBuffer(VulkanContext& ctx) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer      stagingBuffer;
    VmaAllocation stagingAlloc;
    ctx.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingAlloc);

    void* data;
    vmaMapMemory(ctx.allocator, stagingAlloc, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vmaUnmapMemory(ctx.allocator, stagingAlloc);

    ctx.createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vertexBuffer, vertexBufferAlloc);

    ctx.copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);
}

void MeshLoader::createIndexBuffer(VulkanContext& ctx) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    VkBuffer      stagingBuffer;
    VmaAllocation stagingAlloc;
    ctx.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingAlloc);

    void* data;
    vmaMapMemory(ctx.allocator, stagingAlloc, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vmaUnmapMemory(ctx.allocator, stagingAlloc);

    ctx.createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBuffer, indexBufferAlloc);

    ctx.copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);
}
