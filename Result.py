from api.resultAPI import *
from api.app.init import Widget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtWidgets

class AbstractWindow(Widget):
    def on_init(self):
        # create_sphere((0, 0, 0), "Sphere", 200, 20, 20)
        load_model("example.obj", (0, 0, 0), "Yeah")
    
    def update_frame(self):
        # rotate_object("y", "Sphere", 1)
        # rotate_object("x", "Sphere", 1)
        # rotate_object("z", "Sphere", 1)
        rotate_object("y", "Yeah", 1)

        for obj in objects_onscene:
            render_object(obj)
        
        handle_camera_rotation()

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
QSurfaceFormat.setDefaultFormat(fmt)

app = QApplication(sys.argv)
w = AbstractWindow()
set_window_instance(w)
w.show()

sys.exit(app.exec())