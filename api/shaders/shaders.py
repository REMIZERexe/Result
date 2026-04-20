VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

uniform mat4 MVP;

out vec3 fragNormal;
out vec2 fragUV;

void main() {
    gl_Position = vec4(position, 1.0) * MVP;
    fragNormal  = normal;
    fragUV      = uv;
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
in vec3 fragNormal;
in vec2 fragUV;

uniform vec4      objectColor;
uniform vec3      cameraForward;
uniform sampler2D texSampler;
uniform bool      useTexture;
uniform bool      useShading;

// Lighting
uniform bool  useLighting;
uniform int   lightCount;

#define MAX_LIGHTS 16

uniform vec3  lightDir    [MAX_LIGHTS];
uniform vec3  lightColor  [MAX_LIGHTS];
uniform float lightAmbient[MAX_LIGHTS];

out vec4 FragColor;

void main() {
    vec4 base = useTexture ? texture(texSampler, fragUV) : objectColor;

    if (useLighting) {
        vec3 norm   = normalize(fragNormal);
        vec3 result = vec3(0.0);

        for (int i = 0; i < lightCount; i++) {
            vec3 toLight = normalize(-lightDir[i]);
            vec3 ambient = lightAmbient[i] * lightColor[i];
            vec3 diffuse = max(dot(norm, toLight), 0.0) * lightColor[i];
            result += (ambient + diffuse);
        }

        // If no lights, fall back to base color unmodified
        if (lightCount == 0) result = vec3(0.0);

        FragColor = vec4(result * base.rgb, base.a);

    } else if (useShading) {
        float shade = abs(dot(normalize(fragNormal), normalize(cameraForward)));
        shade = 0.25 + shade * 0.75;
        FragColor = vec4(base.rgb * shade, base.a);

    } else {
        FragColor = base;
    }
}
"""