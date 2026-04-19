import sys

sys.dont_write_bytecode = True
from api.resultAPI import *
from api.app.init import App
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

# TODO: Add complex working lighting and lit mode.

class AbstractWindow(App):
    def on_init(self):
        scene = Scene()
        camera = Camera()
        set_scene_instance(scene)
        set_camera_instance(camera)

        camera.Position = [-100, 100, 0]

    def main(self):
        load_texture("grass", os.path.join(get_assets_path(), "assets/textures/grass.jpg"))
        load_texture("bricks", os.path.join(get_assets_path(), "assets/textures/bricks.jpg"))

        create_plane("plane", (0, 0, 0), 1000.0, 1000.0, 512, (0.1, 0.3, 1.0, 1.0))
        set_object_texture("plane", "grass", tiling_x=40.0, tiling_y=40.0)

        create_cube((0, 30, 0), "Cube", 80, 50, 50, (1.0, 1.0, 0, 1.0))
        set_object_texture("Cube", "bricks", tiling_x=2.0, tiling_y=2.0)
        # apply_noise("plane", 1, -20, 60, 80, 0.003, 6, 0.45, 2.0)
        # sync_object_to_gpu("plane")
        create_sphere((0, 100, 0), "ball", 50.0, 50, 50, (0.9, 0.1, 0.3, 1.0))
        load_model(os.path.join(get_assets_path(), "assets/models/v1.gltf"), (100, 100, 0), "V1", (0.0, 0.0, 1.0, 1.0), scale_x=8.0, scale_y=8.0, scale_z=8.0, tex_filter="nearest")
        set_object_rotation("V1", -90, 0, 0)

        self.direction = 1
        self.position_x = 0
        self.limit = 2

    def update_scene(self):
        rotate_object_by("ball", 1.0, 1.0, 1.0)

        speed = 0.02

        self.position_x += speed * self.direction

        if self.position_x > self.limit:
            self.direction = -1
        elif self.position_x < -self.limit:
            self.direction = 1

        move_object(0.0, self.position_x, 0.0, "ball")

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