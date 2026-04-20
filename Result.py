import sys
import os

sys.dont_write_bytecode = True
import api.resultAPI as resultAPI
from api.app.init import App
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

# TODO: Possibility to toggle flat shading
# TODO: Code cleanup
# TODO: Add complex working lighting and lit mode.
# TODO: Start working on edit mode and more settings, windows, buttons, to do stuff in-engine

# TODO: ..... PHYSICS! Source level physics. Like really good realistic physics

class AbstractWindow(App):
    def on_init(self):
        scene = resultAPI.Scene()
        camera = resultAPI.Camera()
        resultAPI.set_scene_instance(scene)
        resultAPI.set_camera_instance(camera)

        camera.Position = [-100, 100, 0]

    def main(self):
        resultAPI.load_texture("grass", os.path.join(resultAPI.get_assets_path(), "assets/textures/grass.jpg"))

        resultAPI.create_plane("plane", (0, 0, 0), 1000.0, 1000.0, 512, (0.1, 0.3, 1.0, 1.0), flat_shading=False)
        resultAPI.set_object_texture("plane", "grass", tiling_x=40.0, tiling_y=40.0)
        # apply_noise("plane", 1, -20, 60, 80, 0.003, 6, 0.45, 2.0)

        # create_cube((0, 30, 0), "Cube", 80, 50, 50, (1.0, 1.0, 0, 1.0))
        # set_object_texture("Cube", "bricks", tiling_x=2.0, tiling_y=2.0)

        # create_sphere((0, 100, 0), "ball", 50.0, 50, 50, (0.9, 0.1, 0.3, 1.0))

        resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/v1.gltf"), (0, 50, 0), "V1", (0.0, 0.0, 1.0, 1.0), tex_filter="nearest")
        resultAPI.set_object_rotation("V1", 0, 90, 0)

        resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/earth_globe.glb"), (0, 80, 90), "earth", (0.0, 1.0, 1.0, 1.0), scale_x=0.08, scale_y=0.08, scale_z=0.08)
        resultAPI.set_object_rotation("earth", 90, 0, 0)

        self.direction = 1
        self.position_x = 0
        self.limit = 2

    def update_scene(self):
        return
        # rotate_object_by("ball", 1.0, 1.0, 1.0)

        # speed = 0.02

        # self.position_x += speed * self.direction

        # if self.position_x > self.limit:
        #     self.direction = -1
        # elif self.position_x < -self.limit:
        #     self.direction = 1

        # move_object(0.0, self.position_x, 0.0, "ball")

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
QSurfaceFormat.setDefaultFormat(fmt)

app = QApplication(sys.argv)
w = AbstractWindow()

resultAPI.set_window_instance(w)

if resultAPI.config.getboolean("window", "fullscreen"):
    w.showFullScreen()
else:
    w.show()

sys.exit(app.exec())