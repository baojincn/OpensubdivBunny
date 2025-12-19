
cmake_minimum_required(VERSION 4.1.1)  # vcpkg 推荐 3.20+
project(OpenSubdivTest LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find vcpkg 提供的依赖 ---
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)



# 注意：OpenSubdiv 默认会构建很多后端（如 CUDA, DX, etc），我们只用 CPU + OpenGL

set(OPENSUBDIV_BUILD_SHARED_LIBS OFF)
set(OPENSUBDIV_BUILD_STATIC_LIBS ON)
set(OPENSUBDIV_BUILD_GPU_TESTS OFF)
set(OPENSUBDIV_BUILD_TUTORIALS OFF)
set(OPENSUBDIV_BUILD_EXAMPLES OFF)
set(OPENSUBDIV_BUILD_DOCS OFF)


set(OPENSUBDIV_CUDA_SUPPORT OFF)
set(OPENSUBDIV_OPENCL_SUPPORT OFF)
set(OPENSUBDIV_DX_SUPPORT OFF)
set(OPENSUBDIV_METAL_SUPPORT OFF)
set(OPENSUBDIV_OPEN_GL_SUPPORT ON)

add_subdirectory(extern/opensubdiv)
include_directories(extern/opensubdiv)

add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${PROJECT_NAME} main.cpp "ResourceManager.h" "ResourceManager.cpp")

# 包含目录（GLM 是 header-only，通过 target_link_libraries 自动处理）
target_include_directories(${PROJECT_NAME} PRIVATE extern/opensubdiv)


# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad::glad        # vcpkg 的 glad 提供此 target
    glm::glm          # vcpkg 的 glm 提供此 target
    assimp::assimp   # vcpkg 的 assimp 提供此 target
    osd_static_cpu
)

# Fix MSVC reporting the wrong standard version
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /Zc:__cplusplus)
endif()