VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

uniform mat4 MVP;

flat out vec3 fragNormal;
out vec2 fragUV;

void main() {
    gl_Position = vec4(position, 1.0) * MVP;
    fragNormal  = normal;
    fragUV      = uv;
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
flat in vec3 fragNormal;
in vec2 fragUV;

uniform vec4      objectColor;
uniform vec3      cameraForward;
uniform sampler2D texSampler;
uniform bool      useTexture;
uniform bool      useShading;

out vec4 FragColor;

void main() {
    vec4 base = useTexture ? texture(texSampler, fragUV) : objectColor;

    if (useShading) {
        float shade = abs(dot(normalize(fragNormal), normalize(cameraForward)));
        shade = 0.25 + shade * 0.75;
        FragColor = vec4(base.rgb * shade, base.a);
    } else {
        FragColor = base;
    }
}
"""