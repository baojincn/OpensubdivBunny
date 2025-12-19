#include <map> // Used for vertex deduplication logic
#include <iostream>
#include "ResourceManager.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


void ResourceManager::RegisterResource(const std::string& name, const std::filesystem::path& path) {
    registeredResources[name] = path;
}

std::shared_ptr<MeshData> ResourceManager::GetMesh(const std::string& name) {
    if (auto it = resourceCache.find(name); it != resourceCache.end()) {
        return it->second;
    }

    auto it = registeredResources.find(name);
	if (it == registeredResources.end()) {
		std::cerr << "[ResourceManager] Error: Failed to load mesh " << name << "' not registered.\n";
		return nullptr;
	}

    std::cout << "[ResourceManager] Loading: " << it->second << " ... ";
    std::shared_ptr<MeshData> mesh = LoadMeshFromFile(it->second);

    if (mesh) {
        resourceCache[name] = mesh;
        std::cout << "Success (" << mesh->vertices.size() << " verts)\n";
    }
    else {
        std::cout << "Failed\n";
    }

    return mesh;
}

std::shared_ptr<MeshData> ResourceManager::LoadMeshFromFile(const std::filesystem::path& path) {
	std::filesystem::path rootPah = PROJECT_ROOT_DIR;
	std::filesystem::path fullPath = rootPah / path;

    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
        aiComponent_NORMALS | aiComponent_TEXCOORDS | aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS);

    const aiScene* scene = importer.ReadFile(fullPath.string(), aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return nullptr;
    }

    auto meshData = std::make_shared<MeshData>();

    // Vertex deduplication helper
    struct Vec3Cmp {
        bool operator()(const glm::vec3& a, const glm::vec3& b) const {
            if (glm::abs(a.x - b.x) > 1e-6f) return a.x < b.x;
            if (glm::abs(a.y - b.y) > 1e-6f) return a.y < b.y;
            return a.z < b.z - 1e-6f;
        }
    };
    std::map<glm::vec3, unsigned int, Vec3Cmp> uniqueVerts;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        const aiMesh* mesh = scene->mMeshes[i];
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            const aiFace& face = mesh->mFaces[j];
            if (face.mNumIndices != 3) continue;

            unsigned int triIdx[3];
            for (int k = 0; k < 3; k++) {
                glm::vec3 pos(mesh->mVertices[face.mIndices[k]].x,
                    mesh->mVertices[face.mIndices[k]].y,
                    mesh->mVertices[face.mIndices[k]].z);

                auto it = uniqueVerts.find(pos);
                if (it == uniqueVerts.end()) {
                    unsigned int newIdx = (unsigned int)meshData->vertices.size();
                    meshData->vertices.push_back(pos);
                    meshData->normals.push_back(glm::vec3(0, 0, 0));
                    meshData->uvs.push_back(glm::vec2(0, 0));
                    uniqueVerts[pos] = newIdx;
                    triIdx[k] = newIdx;
                }
                else {
                    triIdx[k] = it->second;
                }
            }

            if (triIdx[0] == triIdx[1] || triIdx[1] == triIdx[2] || triIdx[2] == triIdx[0]) continue;

            meshData->vertsPerFace.push_back(3);
            meshData->indices.push_back(triIdx[0]);
            meshData->indices.push_back(triIdx[1]);
            meshData->indices.push_back(triIdx[2]);
        }
    }

    // Recalculate Normals
    for (size_t i = 0; i < meshData->indices.size(); i += 3) {
        unsigned int i0 = meshData->indices[i];
        unsigned int i1 = meshData->indices[i + 1];
        unsigned int i2 = meshData->indices[i + 2];
        glm::vec3 edge1 = meshData->vertices[i1] - meshData->vertices[i0];
        glm::vec3 edge2 = meshData->vertices[i2] - meshData->vertices[i0];
        glm::vec3 crossP = glm::cross(edge1, edge2);

        if (glm::length(crossP) > 1e-10f) {
            meshData->normals[i0] += crossP;
            meshData->normals[i1] += crossP;
            meshData->normals[i2] += crossP;
        }
    }
    for (auto& n : meshData->normals) {
        float len = glm::length(n);
        if (len > 1e-10f) n /= len; else n = glm::vec3(0, 1, 0);
    }

    return meshData;
}
