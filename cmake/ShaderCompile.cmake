# compile_shaders(target glsl_files output_dir)
# 将 GLSL 着色器编译为 SPIR-V，输出到 exe 同级的 shaders/ 子目录。
function(compile_shaders TARGET GLSL_FILES OUTPUT_DIR)
    find_program(GLSLC glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    )

    if(NOT GLSLC)
        message(WARNING "[ShaderCompile] 未找到 glslc，跳过着色器编译。"
                        "请安装 Vulkan SDK 并确保 VULKAN_SDK 环境变量已设置。")
        return()
    endif()

    set(SPIRV_FILES "")

    foreach(GLSL ${GLSL_FILES})
        # shader.vert -> vert.spv, shader.frag -> frag.spv
        get_filename_component(FILE_EXT ${GLSL} LAST_EXT)
        string(SUBSTRING ${FILE_EXT} 1 -1 STAGE_NAME)   # 去掉开头的点
        get_filename_component(FILE_NAME ${GLSL} NAME)
        set(SPIRV "${OUTPUT_DIR}/${STAGE_NAME}.spv")

        add_custom_command(
            OUTPUT  ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
            COMMAND ${GLSLC} ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL}
            COMMENT "编译着色器: ${FILE_NAME} -> ${FILE_NAME}.spv"
            VERBATIM
        )

        list(APPEND SPIRV_FILES ${SPIRV})
    endforeach()

    if(SPIRV_FILES)
        add_custom_target(${TARGET}_Shaders ALL DEPENDS ${SPIRV_FILES})
        add_dependencies(${TARGET} ${TARGET}_Shaders)
    endif()
endfunction()
