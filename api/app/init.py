import sys

sys.dont_write_bytecode = True
import numpy

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QCursor
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QMenu, QComboBox, QLineEdit

import api.resultAPI
import api.shaders.shaders
from api.app.widgets import Toolbar
from api.resultAPI import load_texture

class Widget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        api.resultAPI.set_result_instance()
        self.setWindowTitle("Result3D [pre-alpha]")
        self.resize(api.resultAPI.config.getint("window", "width"), api.resultAPI.config.getint("window", "height"))
        self.setMouseTracking(True)

        self.shader = None
        self.vao = None
        self.vbo = None
        self.keys_pressed = set()
        self.initialized = False
        self.on_init()

        # Camera
        self.cam = api.resultAPI.result.MainScene.MainCamera
        self.mouse_sensitivity = api.resultAPI.config.getfloat("camera", "mouse_sensitivity")
        self.velocity = numpy.array([0.0, 0.0, 0.0])
        self.acceleration = api.resultAPI.config.getfloat("camera", "acceleration")
        self.max_speed = api.resultAPI.config.getfloat("camera", "max_speed")
        self.damping = api.resultAPI.config.getfloat("camera", "damping")
        self.last_mouse_pos = None
        self.mouse_captured = False

        # UI Menu
        self.menu_panel = None

        self.fpsTimer = QTimer(self)
        self.tpsTimer = QTimer(self)
        self.fpsTimer.timeout.connect(self.update_frame)
        self.tpsTimer.timeout.connect(self.update_keys)
        self.tpsTimer.timeout.connect(self.update_scene)

        self.ready()

    def initializeGL(self):
        glClearColor(0.0, 0.8, 0.8, 1.0)
        glEnable(GL_DEPTH_TEST)
        self.initShaders()
        self.initBuffers()
        self.initialized = True

    def initShaders(self):
        self.shader = compileProgram(
            compileShader(api.shaders.shaders.VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER),
            compileShader(api.shaders.shaders.FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
        )

    def initBuffers(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

    def paintGL(self):
        if not self.initialized:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        mvp_loc      = glGetUniformLocation(self.shader, "MVP")
        color_loc    = glGetUniformLocation(self.shader, "objectColor")
        fwd_loc      = glGetUniformLocation(self.shader, "cameraForward")
        tex_loc      = glGetUniformLocation(self.shader, "texSampler")
        use_tex_loc  = glGetUniformLocation(self.shader, "useTexture")

        # Camera forward
        yaw_rad   = numpy.radians(self.cam.Yaw)
        pitch_rad = numpy.radians(self.cam.Pitch)
        forward   = numpy.array([
            numpy.cos(pitch_rad) * numpy.cos(yaw_rad),
            numpy.sin(pitch_rad),
            numpy.cos(pitch_rad) * numpy.sin(yaw_rad)
        ], dtype=numpy.float32)
        forward /= numpy.linalg.norm(forward)
        glUniform3f(fwd_loc, *forward)

        # Upload any textures that arrived since last frame
        for tex_name, (data, w, h) in list(api.resultAPI.result.PendingTextures.items()):
            api.resultAPI._upload_texture_to_gpu(tex_name, data, w, h)
            del api.resultAPI.result.PendingTextures[tex_name]

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glUniform1i(tex_loc, 0)   # texture unit 0

        for name, mvp, color in api.resultAPI.result.RenderList:
            if name not in api.resultAPI.result.GPUBuffers:
                obj = next(o for o in api.resultAPI.result.MainScene.ObjectsOnScene
                        if o[0] == name)
                api.resultAPI.upload_object_to_gpu(obj)

            vao_solid, tri_count, vao_wire, edge_count = \
                api.resultAPI.result.GPUBuffers[name]

            glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)

            if self.menu_panel.solid and tri_count > 0:
                glUniform4f(color_loc, *color)

                glBindVertexArray(vao_solid)
                glDrawElements(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None)

            if self.menu_panel.textured:
                # Bind texture if this object has one
                obj      = next(o for o in api.resultAPI.result.MainScene.ObjectsOnScene
                                if o[0] == name)
                tex_name = obj[10] if len(obj) > 10 else None
                if tex_name and tex_name in api.resultAPI.result.Textures:
                    from OpenGL.GL import glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE_2D
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, api.resultAPI.result.Textures[tex_name])
                    glUniform1i(use_tex_loc, 1)
                else:
                    from OpenGL.GL import glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE_2D
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, api.resultAPI.result.Textures["missing"])
                    glUniform1i(use_tex_loc, 1)
                
                glBindVertexArray(vao_solid)
                glDrawElements(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None)

            if self.menu_panel.lit:
                continue

            if self.menu_panel.wireframe:
                glDisable(GL_POLYGON_OFFSET_FILL)
                glUniform1i(use_tex_loc, 0)
                glUniform4f(color_loc, 0.0, 0.0, 0.0, 1.0)
                glBindVertexArray(vao_wire)
                glDrawElements(GL_LINES, edge_count, GL_UNSIGNED_INT, None)
                glEnable(GL_POLYGON_OFFSET_FILL)

        glDisable(GL_POLYGON_OFFSET_FILL)
        glBindVertexArray(0)
        glUseProgram(0)

    def resizeGL(self, width: int, height: int):
        api.resultAPI.config.set("window", "width", str(width))
        api.resultAPI.config.set("window", "height", str(int(width * 9 / 16)))
    
    #! My functions
    def draw_line(self, start, end):
        width, height = self.width(), self.height()
        ndc_start_x = (start[0] / width) * 2 - 1
        ndc_start_y = 1 - (start[1] / height) * 2
        ndc_end_x = (end[0] / width) * 2 - 1
        ndc_end_y = 1 - (end[1] / height) * 2
        
        api.resultAPI.result.EdgeBuffer.append((ndc_start_x, ndc_start_y, ndc_end_x, ndc_end_y))
        self.update()
    #! ----------

    def on_init(self):
        pass

    def main(self):
        pass

    def update_frame(self):
        api.resultAPI.result.EdgeBuffer.clear()
        api.resultAPI.render_scene()

    def update_scene(self):
        pass

    def ready(self):
        if hasattr(api.resultAPI.result, "MainScene") == False:
            print("Result3D: You haven't created any scene! Create a scene instance and set it as the main scene with set_scene_instance()")
            return
        if hasattr(api.resultAPI.result.MainScene, "MainCamera") == False:
            print("Result3D: You haven't created any camera! Create a camera instance and set it as the main camera with set_camera_instance()")
            return
        else:
            matrices = api.resultAPI.Matrices()
            api.resultAPI.set_matrices_instance(matrices)

            self.menu_panel = Toolbar(self)
            self.menu_panel.hide()

            load_texture("missing", "assets/textures/missing.png")

            self.main()
            self.tpsTimer.start(round(1000 / api.resultAPI.config.getint("settings", "tps")))
            self.fpsTimer.start(round(1000 / api.resultAPI.config.getint("settings", "fps")))

    def keyPressEvent(self, e):
        self.keys_pressed.add(e.key())  # add() is idempotent — no duplicates

        if e.key() == Qt.Key.Key_Escape:
            self.toggle_mouse_capture()
        if e.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        if e.key() == Qt.Key.Key_F1:
            self.menu_panel.toggle()

    def keyReleaseEvent(self, e):
        self.keys_pressed.discard(e.key())  # discard() won't throw if missing

    def focusOutEvent(self, event):
        self.keys_pressed.clear()
        self.velocity = numpy.array([0.0, 0.0, 0.0])  # kill drift immediately
        super().focusOutEvent(event)

    def mousePressEvent(self, e):
    # Don't capture mouse if clicking on menu
        if self.menu_panel and self.menu_panel.isVisible() and e.pos().x() < 250:
            return
        
        if not self.mouse_captured:
            self.capture_mouse()

    def mouseMoveEvent(self, e):
        if not self.mouse_captured:
            return
        
        current_pos = e.pos()
        
        if self.last_mouse_pos is None:
            self.last_mouse_pos = current_pos
            return
        
        dx = current_pos.x() - self.last_mouse_pos.x()
        dy = current_pos.y() - self.last_mouse_pos.y()

        self.cam.Yaw += dx * self.mouse_sensitivity
        self.cam.Pitch -= dy * self.mouse_sensitivity

        self.cam.Pitch = max(-89.0, min(89.0, self.cam.Pitch))

        center = QPoint(self.width() // 2, self.height() // 2)
        QCursor.setPos(self.mapToGlobal(center))
        self.last_mouse_pos = center
    
    def capture_mouse(self):
        self.mouse_captured = True
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setFocus()

        center = QPoint(self.width() // 2, self.height() // 2)
        QCursor.setPos(self.mapToGlobal(center))
        self.last_mouse_pos = center
    
    def release_mouse(self):
        self.mouse_captured = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.last_mouse_pos = None
    
    def toggle_mouse_capture(self):
        if self.mouse_captured:
            self.release_mouse()
        else:
            self.capture_mouse()
    
    def get_camera_direction(self):
        yaw_rad = numpy.radians(self.cam.Yaw)

        forward_movement = numpy.array([
            numpy.cos(yaw_rad),
            0.0,
            numpy.sin(yaw_rad)
        ])
        forward_movement = forward_movement / numpy.linalg.norm(forward_movement)
        
        world_up = numpy.array([0.0, 1.0, 0.0])
        right = numpy.cross(forward_movement, world_up)
        right = right / numpy.linalg.norm(right)
        
        return forward_movement, right, world_up
    
    def update_keys(self):
        forward, right, up = self.get_camera_direction()
        input_direction = numpy.array([0.0, 0.0, 0.0])
        
        if Qt.Key.Key_Z in self.keys_pressed:
            input_direction += forward  # forward
        
        if Qt.Key.Key_S in self.keys_pressed:
            input_direction -= forward  # backward
        
        if Qt.Key.Key_Q in self.keys_pressed:
            input_direction -= right  # left
        
        if Qt.Key.Key_D in self.keys_pressed:
            input_direction += right  # right
        
        if Qt.Key.Key_Space in self.keys_pressed:
            input_direction += up  # up
        
        if Qt.Key.Key_Shift in self.keys_pressed:
            input_direction -= up  # down
        
        if numpy.any(input_direction != 0):
            input_direction = input_direction / numpy.linalg.norm(input_direction)
        self.velocity += input_direction * self.acceleration
        
        speed = numpy.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        self.velocity *= self.damping
        self.cam.Position += self.velocity
        
        self.update()