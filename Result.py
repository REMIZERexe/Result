import sys

sys.dont_write_bytecode = True
from api.resultAPI import *
from api.app.init import Widget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

# TODO: Fix load_model() not reading verticies with "e".
# TODO: Add vertices post-processing for rendering in camera view.

class AbstractWindow(Widget):
    def on_init(self):
        scene = Scene()
        camera = Camera()
        set_scene_instance(scene)
        set_camera_instance(camera)

        camera.Position = [-100, 0, 0]

    def main(self): 
        create_cone((0, 0, 0), "Cone", 40, 50, 4, False)

    def update_scene(self):
        rotate_object("y", "Cone", 1.0)

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