import sys
import os

sys.dont_write_bytecode = True
import api.resultAPI as resultAPI
from api.app.init import App
from PyQt6.QtGui import QSurfaceFormat, QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

# TODO: Add complex working lighting and lit mode.

# TODO: Fix textures not falling back to missing when none.
# TODO: Fix camera speed not working
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
        self.sun = resultAPI.create_directional_light(
            direction=(0.4, -1.0, 0.6),
            color=(1.0, 1.0, 0.5),
            ambient=0.15
        )

        resultAPI.load_texture("grass", os.path.join(resultAPI.get_assets_path(), "assets/textures/grass.jpg"))

        resultAPI.create_plane("plane", (0, 0, 0), 1000.0, 1000.0, color=(0.1, 0.3, 1.0, 1.0), flat_shading=False)
        resultAPI.set_object_texture("plane", "grass", tiling_x=40.0, tiling_y=40.0)
        # apply_noise("plane", 1, -20, 60, 80, 0.003, 6, 0.45, 2.0)

        # create_cube((0, 30, 0), "Cube", 80, 50, 50, (1.0, 1.0, 0, 1.0))
        # set_object_texture("Cube", "bricks", tiling_x=2.0, tiling_y=2.0)

        # create_sphere((0, 100, 0), "ball", 50.0, 50, 50, (0.9, 0.1, 0.3, 1.0))

        # resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/v1.gltf"), (0, 50, 0), "V1", (0.0, 0.0, 1.0, 1.0), tex_filter="nearest")
        # resultAPI.set_object_rotation("V1", 0, 90, 0)

        # resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/earth_globe.glb"), (0, 80, 90), "earth", (0.0, 1.0, 1.0, 1.0), scale_x=0.08, scale_y=0.08, scale_z=0.08)
        # resultAPI.set_object_rotation("earth", 90, 0, 0)

        # resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/earth_globe.glb"), (0, 160, 90), "earth", (0.0, 1.0, 1.0, 1.0), scale_x=0.08, scale_y=0.08, scale_z=0.08)
        # resultAPI.set_object_rotation("earth", 90, 0, 0)

        # resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/earth_globe.glb"), (0, 160, 180), "earth", (0.0, 1.0, 1.0, 1.0), scale_x=0.08, scale_y=0.08, scale_z=0.08)
        # resultAPI.set_object_rotation("earth", 90, 0, 0)

        # resultAPI.load_model(os.path.join(resultAPI.get_assets_path(), "assets/models/earth_globe.glb"), (90, 80, 90), "earth", (0.0, 1.0, 1.0, 1.0), scale_x=0.08, scale_y=0.08, scale_z=0.08)
        # resultAPI.set_object_rotation("earth", 90, 0, 0)

        self.direction = 1
        self.position_x = 0
        self.limit = 2

    def update_scene(self):
        return
        # rotate_object_by("ball", 1.0, 1.0, 1.0)

        # speed = 0.2

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

font_id = QFontDatabase.addApplicationFont("assets/fonts/W95F.otf")
family = QFontDatabase.applicationFontFamilies(font_id)[0]

font = QFont(family, 10)
font.setStyleStrategy(QFont.StyleStrategy.NoSubpixelAntialias)

app.setFont(font)

w = AbstractWindow()

resultAPI.set_window_instance(w)

if resultAPI.config.getboolean("window", "fullscreen"):
    w.showFullScreen()
else:
    w.show()

sys.exit(app.exec())