import sys
sys.dont_write_bytecode = True
import numpy as np

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PyQt6.QtCore import QTimer
from PyQt6 import QtGui

from api.resultAPI import *
from api.shaders.shaders import *

res = Result()

class Widget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Result3D [pre-alpha]")
        set_window_settings_instance(WindowSettings())
        self.resize(Result.WindowParam.WindowSize["width"], Result.WindowParam.WindowSize["height"])
        self.setMouseTracking(True)

        self.shader = None
        self.vao = None
        self.vbo = None
        self.on_init()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.ready()

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

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        
        for line in res.EdgeBuffer:
            vertices = np.array([
                line[0], line[1],
                line[2], line[3]
            ], dtype=np.float32)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_LINES, 0, 2)
        
        res.EdgeBuffer.clear()

        glBindVertexArray(0)
        glUseProgram(0)
        glFinish()

    def resizeGL(self, width: int, height: int):
        Result.WindowParam.WindowSize = {
            "width": width,
            "height": int(width * 9 / 16)
        }
    
    #! My functions
    def draw_line(self, start, end):
        width, height = self.width(), self.height()
        ndc_start_x = (start[0] / width) * 2 - 1
        ndc_start_y = 1 - (start[1] / height) * 2
        ndc_end_x = (end[0] / width) * 2 - 1
        ndc_end_y = 1 - (end[1] / height) * 2
        
        res.EdgeBuffer.append((ndc_start_x, ndc_start_y, ndc_end_x, ndc_end_y))
        self.update()
    #! ----------

    def on_init(self):
        pass

    def main(self):
        pass

    def update_frame(self):
        render_scene()

    def ready(self):
        if hasattr(Result, "MainScene") == False:
            print("Result3D: You haven't created any scene! Create a scene instance and set it as the main scene with set_scene_instance()")
            return
        if hasattr(Result.MainScene, "MainCamera") == False:
            print("Result3D: You haven't created any camera! Create a camera instance and set it as the main camera with set_camera_instance()")
            return
        else:
            matrices = Matrices()
            set_matrices_instance(matrices)

            self.main()
            self.timer.start(16)