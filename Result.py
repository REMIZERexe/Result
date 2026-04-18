import sys

sys.dont_write_bytecode = True
from api.resultAPI import *
from api.app.init import Widget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

# BUGS:
# - Fix load_model() not reading verticies with "e".

# TODO: Add complex working lighting and lit mode.

class AbstractWindow(Widget):
    def on_init(self):
        scene = Scene()
        camera = Camera()
        set_scene_instance(scene)
        set_camera_instance(camera)

        camera.Position = [-100, 0, 0]

    def main(self):
        load_texture("grass", "assets/textures/grass.jpg")
        load_texture("bricks", "assets/textures/bricks.jpg")

        create_plane("plane", (0, 0, 0), 1000.0, 1000.0, 512, (0.1, 0.3, 1.0, 1.0))
        set_object_texture("plane", "grass")

        create_cube((0, 30, 0), "Cube", 50, 50, 50, (1.0, 1.0, 0, 1.0))
        set_object_texture("Cube", "bricks")
        apply_noise("plane", 1, -20, 60, 80, 0.003, 6, 0.45, 2.0)
        sync_object_to_gpu("plane")
        create_sphere((0, 100, 0), "ball", 50.0, 50, 50, (0.9, 0.1, 0.3, 1.0))
        load_model("assets/models/v1.obj", (100, 100, 0), "V1", (0.0, 0.0, 1.0, 1.0))

    def update_scene(self):
        return

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
QSurfaceFormat.setDefaultFormat(fmt)

app = QApplication(sys.argv)
w = AbstractWindow()

set_window_instance(w)

if config.getboolean("window", "fullscreen"):
    w.showFullScreen()
else:
    w.show()

sys.exit(app.exec())