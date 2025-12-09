#define _USE_MATH_DEFINES

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/far/primvarRefiner.h>

typedef struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    void Clear(void* = 0) { pos = glm::vec3(0); normal = glm::vec3(0); uv = glm::vec2(0); }
    void AddWithWeight(const Vertex& src, float weight) { pos += src.pos * weight; normal += src.normal * weight; uv += src.uv * weight; }
} Vertex;

struct BaseMesh
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;
    std::vector<int> vertsPerFace;
};


BaseMesh g_baseMesh;
std::vector<Vertex> g_renderVerts;
std::vector<unsigned int> g_renderIndices;

GLuint g_vao, g_vbo, g_ebo;
int g_currentLevel = 0; // Subdivision level (0~5)

bool g_showWireframe = false;


int g_modelIndex = 0;   // 0: Bunny, 1: Suzanne, 2:original_bunny, 3: Cube
const int MAX_MODELS = 4;

bool g_leftMouseDown = false;
bool g_rightMouseDown = false;
double g_lastX = 0.0f;
double g_lastY = 0.0f;
float g_rotX = 0.0f;
float g_rotY = 0.0f;
float g_cameraDist = 3.0f;

void updateMeshSubdivsion(int level);
void updateBuffers();
bool loadOBJ(const std::string& path, BaseMesh& baseMesh);
void createCube(BaseMesh& mesh);
void loadModelData(int index);

// Testing cube
void createCube(BaseMesh& mesh) {
    mesh.vertices.clear(); mesh.normals.clear(); mesh.uvs.clear(); 
    mesh.indices.clear(); mesh.vertsPerFace.clear();

    std::vector<glm::vec3> p = {
        {-0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
        {-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}
    };
    std::vector<unsigned int> idx = {
        0,1,2, 2,3,0,  1,5,6, 6,2,1,  5,4,7, 7,6,5,
        4,0,3, 3,7,4,  3,2,6, 6,7,3,  4,5,1, 1,0,4
    };

    mesh.vertices = p;
    mesh.indices = idx;
    for(auto& v : p) { mesh.normals.push_back(glm::vec3(0,0,0)); mesh.uvs.push_back(glm::vec2(0,0)); }
    for(int i=0; i<12; ++i) mesh.vertsPerFace.push_back(3);

    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        unsigned int i0 = mesh.indices[i];
        unsigned int i1 = mesh.indices[i+1];
        unsigned int i2 = mesh.indices[i+2];
        glm::vec3 v0 = mesh.vertices[i0];
        glm::vec3 v1 = mesh.vertices[i1];
        glm::vec3 v2 = mesh.vertices[i2];
        glm::vec3 crossP = glm::cross(v1 - v0, v2 - v0);
        if (glm::length(crossP) > 1e-10f) {
            mesh.normals[i0] += crossP; mesh.normals[i1] += crossP; mesh.normals[i2] += crossP;
        }
    }
    for (auto& n : mesh.normals) if(glm::length(n)>0) n = glm::normalize(n);
}

bool loadOBJ(const std::string& path, BaseMesh& baseMesh) {
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, 
        aiComponent_NORMALS | aiComponent_TEXCOORDS | aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS);

    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return false;
    }

    baseMesh.vertices.clear(); baseMesh.normals.clear();
    baseMesh.uvs.clear(); baseMesh.indices.clear(); baseMesh.vertsPerFace.clear();

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
                    unsigned int newIdx = (unsigned int)baseMesh.vertices.size();
                    baseMesh.vertices.push_back(pos);
                    baseMesh.normals.push_back(glm::vec3(0,0,0)); 
                    baseMesh.uvs.push_back(glm::vec2(0,0));
                    uniqueVerts[pos] = newIdx;
                    triIdx[k] = newIdx;
                } else {
                    triIdx[k] = it->second;
                }
            }

            if (triIdx[0] == triIdx[1] || triIdx[1] == triIdx[2] || triIdx[2] == triIdx[0]) continue;

            baseMesh.vertsPerFace.push_back(3);
            baseMesh.indices.push_back(triIdx[0]);
            baseMesh.indices.push_back(triIdx[1]);
            baseMesh.indices.push_back(triIdx[2]);
        }
    }

    // Calculate base mesh normal
    for (size_t i = 0; i < baseMesh.indices.size(); i += 3) {
        unsigned int i0 = baseMesh.indices[i];
        unsigned int i1 = baseMesh.indices[i+1];
        unsigned int i2 = baseMesh.indices[i+2];
        glm::vec3 edge1 = baseMesh.vertices[i1] - baseMesh.vertices[i0];
        glm::vec3 edge2 = baseMesh.vertices[i2] - baseMesh.vertices[i0];
        glm::vec3 crossP = glm::cross(edge1, edge2);

        if (glm::length(crossP) > 1e-10f) {
            baseMesh.normals[i0] += crossP; baseMesh.normals[i1] += crossP; baseMesh.normals[i2] += crossP;
        }
    }
    for (auto& n : baseMesh.normals) {
        float len = glm::length(n);
        if (len > 1e-10f) n /= len; else n = glm::vec3(0, 1, 0);
    }

    std::cout << "Model Loaded (" << path << "): " << baseMesh.vertices.size() << " verts." << std::endl;
    return true;
}

// Load models
void loadModelData(int index) {
    g_baseMesh.vertices.clear(); g_baseMesh.normals.clear(); 
    g_baseMesh.uvs.clear(); g_baseMesh.indices.clear(); g_baseMesh.vertsPerFace.clear();

    if (index == 0) {
        loadOBJ("../bunny.obj", g_baseMesh);
    } 
    else if (index == 1) {
        loadOBJ("../suzanne.obj", g_baseMesh);
    }
    else if (index == 2) {
        loadOBJ("../original_bunny.obj", g_baseMesh);
    }
    else {
        createCube(g_baseMesh);
        std::cout << "Model Loaded: Cube" << std::endl;
    }

    g_currentLevel = 0;
    updateMeshSubdivsion(0);
}

void updateMeshSubdivsion(int level)
{
    using namespace OpenSubdiv;

    g_renderVerts.clear();
    g_renderIndices.clear();

    // Level 0:  BaseMesh
    if (level <= 0) {
        for (size_t i = 0; i < g_baseMesh.vertices.size(); ++i) {
            Vertex vert;
            vert.pos = g_baseMesh.vertices[i];
            vert.normal = g_baseMesh.normals[i]; 
            vert.uv = g_baseMesh.uvs[i];
            g_renderVerts.push_back(vert);
        }
        g_renderIndices = g_baseMesh.indices;
        updateBuffers(); 
        return;
    }

    // Setup Topology
    Far::TopologyDescriptor desc;
    desc.numVertices = (int)g_baseMesh.vertices.size();
    desc.numFaces = (int)g_baseMesh.vertsPerFace.size();
    desc.numVertsPerFace = g_baseMesh.vertsPerFace.data();
    desc.vertIndicesPerFace = (Far::Index*)g_baseMesh.indices.data();

    Sdc::SchemeType type = Sdc::SchemeType::SCHEME_LOOP;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc,
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options(type, options));

    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(level));

    int numTotalVertices = refiner->GetNumVerticesTotal();
    std::vector<Vertex> refinedVerts(numTotalVertices);

    // Fill Level 0
    int numLevel0 = g_baseMesh.vertices.size();
    for (int i = 0; i < numLevel0; ++i) {
        refinedVerts[i].pos = g_baseMesh.vertices[i];
        refinedVerts[i].normal = glm::vec3(0,0,0); 
        refinedVerts[i].uv = glm::vec2(0,0);
    }

    Far::PrimvarRefiner primvarRefiner(*refiner);
    
    // Interpolate with Offsets
    int srcOffset = 0;
    int dstOffset = numLevel0; 

    for (int i = 1; i <= level; ++i) {
        Vertex* srcPtr = refinedVerts.data() + srcOffset;
        Vertex* dstPtr = refinedVerts.data() + dstOffset;
        primvarRefiner.Interpolate(i, srcPtr, dstPtr);
        srcOffset = dstOffset;
        dstOffset += refiner->GetLevel(i).GetNumVertices();
    }

    // Extract Indices with Offset
    Far::TopologyLevel const &lastLevel = refiner->GetLevel(level);
    int numFaces = lastLevel.GetNumFaces();
    int firstVertexOffset = srcOffset; 

    for (int face = 0; face < numFaces; ++face) {
        Far::ConstIndexArray faceVerts = lastLevel.GetFaceVertices(face);
        if (faceVerts.size() != 3) continue;
        g_renderIndices.push_back(faceVerts[0] + firstVertexOffset);
        g_renderIndices.push_back(faceVerts[1] + firstVertexOffset);
        g_renderIndices.push_back(faceVerts[2] + firstVertexOffset);
    }

    // Recalculate Normals
    for(auto& v : refinedVerts) v.normal = glm::vec3(0.0f);

    for (size_t i = 0; i < g_renderIndices.size(); i += 3) {
        unsigned int idx0 = g_renderIndices[i];
        unsigned int idx1 = g_renderIndices[i+1];
        unsigned int idx2 = g_renderIndices[i+2];

        if (idx0 >= refinedVerts.size() || idx1 >= refinedVerts.size() || idx2 >= refinedVerts.size()) continue;

        glm::vec3 v0 = refinedVerts[idx0].pos;
        glm::vec3 v1 = refinedVerts[idx1].pos;
        glm::vec3 v2 = refinedVerts[idx2].pos;

        glm::vec3 crossP = glm::cross(v1 - v0, v2 - v0);
        if (glm::length(crossP) > 1e-10f) {
            refinedVerts[idx0].normal += crossP;
            refinedVerts[idx1].normal += crossP;
            refinedVerts[idx2].normal += crossP;
        }
    }

    for(auto& v : refinedVerts) {
        float len = glm::length(v.normal);
        if (len > 1e-10f) v.normal /= len;
        else v.normal = glm::vec3(0,1,0);
    }

    g_renderVerts = refinedVerts;
    delete refiner;
    updateBuffers();
}

void updateBuffers()
{
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, g_renderVerts.size() * sizeof(Vertex), g_renderVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_renderIndices.size() * sizeof(unsigned int), g_renderIndices.data(), GL_STATIC_DRAW);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (action == GLFW_PRESS)
    {
        // Appy wireframe (Press '0')
        if (key == GLFW_KEY_0) {
            g_showWireframe = !g_showWireframe;
            std::cout << "Wireframe Mode: " << (g_showWireframe ? "ON" : "OFF") << std::endl;
        }

        // Change model (+/-)
        bool modelChanged = false;
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
            g_modelIndex = (g_modelIndex + 1) % MAX_MODELS;
            modelChanged = true;
        }
        else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
            g_modelIndex = (g_modelIndex - 1 + MAX_MODELS) % MAX_MODELS;
            modelChanged = true;
        }

        if (modelChanged) {
            loadModelData(g_modelIndex);
        }

        // Change subvision level (1-6)
        int newLevel = -1;
        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6) {
            newLevel = key - GLFW_KEY_1;
        }

        if (newLevel != -1 && newLevel != g_currentLevel) {
            g_currentLevel = newLevel;
            updateMeshSubdivsion(g_currentLevel);
        }

        std::string modelName = (g_modelIndex == 0) ? "Bunny" : (g_modelIndex == 1 ? "Suzanne" : "Cube");
        std::string title = modelName + " | Level: " + std::to_string(g_currentLevel + 1) + 
                            " | Tris: " + std::to_string(g_renderIndices.size()/3) +
                            (g_showWireframe ? " | Wireframe" : "");
        glfwSetWindowTitle(window, title.c_str());
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        g_lastX = xpos; g_lastY = ypos;
        if (button == GLFW_MOUSE_BUTTON_LEFT) g_leftMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) g_rightMouseDown = true;
    }
    else if (action == GLFW_RELEASE) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) g_leftMouseDown = false;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) g_rightMouseDown = false;
    }
}

void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    double dx = xpos - g_lastX;
    double dy = ypos - g_lastY;
    if (g_leftMouseDown) {
        float sensitivity = 0.01f;
        g_rotY += (float)dx * sensitivity;
        g_rotX += (float)dy * sensitivity;
    }
    if (g_rightMouseDown) {
        float zoomSensitivity = 0.05f;
        g_cameraDist += (float)dy * zoomSensitivity;
        if (g_cameraDist < 0.1f) g_cameraDist = 0.1f;
        if (g_cameraDist > 20.0f) g_cameraDist = 20.0f;
    }
    g_lastX = xpos; g_lastY = ypos;
}

static const char *vertex_shader_text =
    "#version 330 core\n"
    "layout(location = 0) in vec3 vPos;\n"
    "layout(location = 1) in vec3 vNormal;\n"
    "layout(location = 2) in vec2 vTexCoord;\n"
    "uniform mat4 MVP;\n"
    "uniform mat4 ModelMatrix;\n"
    "out vec3 WorldNormal;\n"
    "out vec2 TexCoord;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = MVP * vec4(vPos, 1.0);\n"
    "    WorldNormal = mat3(transpose(inverse(ModelMatrix))) * vNormal;\n"
    "    TexCoord = vTexCoord;\n"
    "}\n";

static const char *fragment_shader_text =
    "#version 330 core\n"
    "in vec3 WorldNormal;\n"
    "in vec2 TexCoord;\n"
    "uniform vec3 LightDirection;\n"
    "uniform vec3 AmbientColor;\n"
    "uniform vec3 DiffuseColor;\n"
    "out vec4 FragmentColor;\n"
    "void main()\n"
    "{\n"
    "   vec3 BaseColor = vec3(0.8, 0.8, 0.8);\n"
    "   vec3 N = normalize(WorldNormal);\n"
    "   vec3 L = normalize(LightDirection);\n"
    "   float NdotL = max(dot(N, L), 0.0);\n"
    "   vec3 Lighting = AmbientColor + DiffuseColor * NdotL;\n"
    "   vec3 ViewDir = vec3(0, 0, 1);\n"
    "   vec3 H = normalize(L + ViewDir);\n"
    "   float NdotH = max(dot(N, H), 0.0);\n"
    "   float spec = pow(NdotH, 64.0);\n"
    "   FragmentColor = vec4(BaseColor * Lighting + vec3(0.4) * spec, 1.0);\n"
    "}\n";

static void error_callback(int error, const char *description) { fprintf(stderr, "Error: %s\n", description); }

int main(void)
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1024, 768, "OpenSubdiv Viewer", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;
    glfwSwapInterval(1);

    auto compileShader = [](GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, NULL);
        glCompileShader(shader);
        GLint success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success) { char info[512]; glGetShaderInfoLog(shader, 512, NULL, info); std::cout << "Shader Error: " << info << std::endl; }
        return shader;
    };
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertex_shader_text);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragment_shader_text);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glGenBuffers(1, &g_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, uv));

    // Default model 0 (Bunny)
    loadModelData(0);

    const GLint mvpLocation = glGetUniformLocation(program, "MVP");
    const GLint modelMatrixLocation = glGetUniformLocation(program, "ModelMatrix");
    const GLint light_dir_location = glGetUniformLocation(program, "LightDirection");
    const GLint ambient_color_location = glGetUniformLocation(program, "AmbientColor");
    const GLint diffuse_color_location = glGetUniformLocation(program, "DiffuseColor");

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window))
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float ratio = width / (float)(height > 0 ? height : 1);

        glViewport(0, 0, width, height);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (g_showWireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glm::mat4 m = glm::mat4(1.0f);
        m = glm::rotate(m, g_rotX, glm::vec3(1.0f, 0.0f, 0.0f));
        m = glm::rotate(m, g_rotY, glm::vec3(0.0f, 1.0f, 0.0f));
        m = glm::scale(m, glm::vec3(4.0f));

        glm::mat4 v = glm::lookAt(glm::vec3(0.0f, 1.0f, g_cameraDist), glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 p = glm::perspective(glm::radians(45.0f), ratio, 0.1f, 100.0f);
        glm::mat4 mvp = p * v * m;

        glUseProgram(program);
        glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, (const GLfloat *)&mvp);
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, (const GLfloat *)&m);
        glUniform3f(ambient_color_location, 0.2f, 0.2f, 0.2f);
        glUniform3f(diffuse_color_location, 0.8f, 0.8f, 0.8f);
        glm::vec3 LightDir = glm::normalize(glm::vec3(0.5f, 0.5f, 1.0f));
        glUniform3fv(light_dir_location, 1, (const GLfloat *)&LightDir);

        glBindVertexArray(g_vao);
        if (!g_renderIndices.empty()) {
            glDrawElements(GL_TRIANGLES, (GLsizei)g_renderIndices.size(), GL_UNSIGNED_INT, 0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}