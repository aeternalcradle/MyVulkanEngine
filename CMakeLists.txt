cmake_minimum_required(VERSION 3.20)
project(MyVulkanEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------
# 依赖：FetchContent 自动下载 GLFW 和 GLM
# ---------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# ---------------------------------------------------------------
# 依赖：Vulkan SDK（已安装，直接 find_package）
# ---------------------------------------------------------------
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------
# 源文件
# ---------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/Application.cpp
    src/core/Window.cpp
    src/rhi/VulkanContext.cpp
    src/rhi/SwapChain.cpp
    src/rhi/Image.cpp
    src/renderer/Pipeline.cpp
    src/renderer/Renderer.cpp
    src/renderer/ShadowMap.cpp
    src/resource/TextureManager.cpp
    src/resource/MeshLoader.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# ---------------------------------------------------------------
# Include 路径
# ---------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src        # 保持 #include "rhi/..." 等路径有效
    ${CMAKE_SOURCE_DIR}/external   # stb_image.h / tiny_obj_loader.h
)

# ---------------------------------------------------------------
# 链接库
# ---------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

# ---------------------------------------------------------------
# 着色器编译（GLSL -> SPIR-V）
# ---------------------------------------------------------------
include(cmake/ShaderCompile.cmake)

file(GLOB GLSL_FILES
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp"
)

# 第一步：编译到固定中间目录
set(SPIRV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})
compile_shaders(${PROJECT_NAME} "${GLSL_FILES}" ${SPIRV_OUTPUT_DIR})

# 第二步：post-build 把 spv 复制到 exe 同级的 shaders/ 目录
# （兼容 MSVC 多配置 Debug/Release，$<TARGET_FILE_DIR:...> 在 COMMAND 中有效）
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SPIRV_OUTPUT_DIR}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "复制 SPIR-V 着色器到输出目录"
)

# post-build 把 assets/ 复制到 exe 同级目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "复制 assets 到输出目录"
)

# ---------------------------------------------------------------
# 编译选项
# ---------------------------------------------------------------
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /utf-8)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)
