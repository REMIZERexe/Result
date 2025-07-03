import sys
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PyQt6.QtCore import QTimer


VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

dx = 0
dy = 0

class Widget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Result3D [pre-alpha]")
        self.resize(1280, 720)
        self.setMouseTracking(True)

        self.shader = None
        self.vao = None
        self.vbo = None
        self.lines = []
        self.on_init()

        self.last_mouse_pos = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)  # вызывает paintGL()
        self.timer.start(16)  # ~60 FPS (1000 ms / 60 ≈ 16)

    def initializeGL(self):
        glClearColor(0, 0.8, 0.8, 1)
        self.initShaders()
        self.initBuffers()

    def initShaders(self):
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
        )

    def initBuffers(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

    def draw_line(self, start, end):
        width, height = self.width(), self.height()
        ndc_start_x = (start[0] / width) * 2 - 1
        ndc_start_y = 1 - (start[1] / height) * 2
        ndc_end_x = (end[0] / width) * 2 - 1
        ndc_end_y = 1 - (end[1] / height) * 2
        
        self.lines.append((ndc_start_x, ndc_start_y, ndc_end_x, ndc_end_y))
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        self.update_frame()
        
        for line in self.lines:
            vertices = np.array([
                line[0], line[1],
                line[2], line[3]
            ], dtype=np.float32)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_LINES, 0, 2)
        
        self.lines.clear()

        glBindVertexArray(0)
        glUseProgram(0)
        glFinish()

    def resizeGL(self, width: int, height: int):
        global centerX, centerY
        centerX, centerY = width // 2, height // 2
    
    def mouseMoveEvent(self, event):
        global dx, dy
        pos = event.position()
        x, y = pos.x(), pos.y()
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            return
        else: self.last_mouse_pos = (x, y)

        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        self.last_mouse_pos = (x, y)
    
    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None
    
    def on_init(self):
        return

    def update_frame(self):
        return