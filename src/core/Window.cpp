#include "core/Window.h"
#include "rhi/VulkanTypes.h"

Window::Window() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

Window::~Window() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Window::framebufferResizeCallback(GLFWwindow* win, int /*width*/, int /*height*/) {
    auto self = reinterpret_cast<Window*>(glfwGetWindowUserPointer(win));
    self->framebufferResized = true;
}
