#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>

// 封装 GLFW 窗口生命周期与 framebuffer resize 事件
class Window {
public:
    Window();
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    GLFWwindow* getHandle()   const { return window; }
    bool shouldClose()        const { return glfwWindowShouldClose(window); }
    bool wasResized()         const { return framebufferResized; }
    void resetResized()             { framebufferResized = false; }

private:
    GLFWwindow* window           = nullptr;
    bool        framebufferResized = false;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};
