#version 450

layout(location = 0) in vec3 viewNormal;

layout(location = 0) out vec4 outNormal;

void main() {
    outNormal = vec4(normalize(viewNormal), 0.0);
}
