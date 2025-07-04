import sys
sys.dont_write_bytecode = True
from api.resultAPI import *
from api.app.init import Widget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

# TODO: Rewrite the camera movement and rotation, but now for PyQT6.
# TODO: Fix load_model() not reading verticies with "e".

class AbstractWindow(Widget):
    def on_init(self): pass

    def main(self): pass

    def update_frame(self):
        super().update_frame()

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
QSurfaceFormat.setDefaultFormat(fmt)

app = QApplication(sys.argv)
w = AbstractWindow()

set_window_instance(w)

if Result.WindowParam.Fullscreen:
    w.showFullScreen()
else:
    w.show()

sys.exit(app.exec())