#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include <glm/glm.hpp>

struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;
    std::vector<int> vertsPerFace;
};

class ResourceManager {
public:
    ResourceManager() = default;
	// Disable copy constructor and assignment operator
    ResourceManager(const ResourceManager&) = delete;
	ResourceManager& operator=(const ResourceManager&) = delete;

    void RegisterResource(const std::string& name, const std::filesystem::path& path);

    [[nodiscard]] std::shared_ptr<MeshData> GetMesh(const std::string& name);

private:
    std::shared_ptr<MeshData> LoadMeshFromFile(const std::filesystem::path& path);
	std::unordered_map<std::string, std::filesystem::path> registeredResources;
    std::unordered_map<std::string, std::shared_ptr<MeshData>> resourceCache;
};

