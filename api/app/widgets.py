from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QMenu, QComboBox, QLineEdit, QHBoxLayout
from PyQt6.QtCore import Qt
import numpy

import api.resultAPI

class Toolbar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = api.resultAPI.result.MainScene.MainCamera
        self.mouse_sensitivity = api.resultAPI.config.getfloat("camera", "mouse_sensitivity")
        self.velocity = numpy.array([0.0, 0.0, 0.0])
        self.acceleration = api.resultAPI.config.getfloat("camera", "acceleration")
        self.max_speed = api.resultAPI.config.getfloat("camera", "max_speed")
        self.damping = api.resultAPI.config.getfloat("camera", "damping")

        # Apply styles to self, not a child widget
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 200);
                border-right: 2px solid rgba(100, 0, 100, 255);
                margin: 3px;
            }
            QLabel {
                color: white;
                font-size: 14px;
                padding: 5px;
            }
            QPushButton {
                background-color: rgba(100, 0, 100, 255);
                color: white;
                border: 1px solid rgba(100, 0, 100, 255);
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 255); 
            }
            QSlider::groove:horizontal {
                margin: 3px;
                background: rgba(60, 60, 60, 255);
            }
            QSlider::handle:horizontal {
                background: rgba(100, 0, 100, 255);
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        # Create layout directly on self
        menu_layout = QVBoxLayout(self)
        menu_layout.addSpacing(20)
        menu_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title = QLabel("Result3D Menu")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        menu_layout.addWidget(title)
        
        # Camera
        cam_label = QLabel("Camera")
        cam_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        menu_layout.addWidget(cam_label)

        speed_label = QLabel(f"Speed: {self.max_speed:.1f}")
        menu_layout.addWidget(speed_label)
        speed_slider = QSlider(Qt.Orientation.Horizontal)
        speed_slider.setMinimum(1)
        speed_slider.setMaximum(100)
        speed_slider.setValue(int(self.max_speed * 10))
        speed_slider.valueChanged.connect(lambda v: self.update_speed(v, speed_label))
        menu_layout.addWidget(speed_slider)

        # Objects
        obj_label = QLabel("Objects")
        obj_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        menu_layout.addWidget(obj_label)

        self.add_drop = QComboBox()
        self.add_drop.setPlaceholderText("Select Object")
        self.add_drop.addItems(["Cube", "Sphere", "Cone"])
        self.add_drop.currentTextChanged.connect(self.update_sizes)
        menu_layout.addWidget(self.add_drop)

        position = QWidget()
        position.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
            QLineEdit {
                background: #222;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit::placeholder {
                color: #888;
            }
        """)
        position_layout = QHBoxLayout(position)
        position_layout.setContentsMargins(0, 0, 0, 0)
        x = QLineEdit()
        x.setPlaceholderText("X")
        y = QLineEdit()
        y.setPlaceholderText("Y")
        z = QLineEdit()
        z.setPlaceholderText("Z")
        position_layout.addWidget(x)
        position_layout.addWidget(y)
        position_layout.addWidget(z)
        menu_layout.addWidget(position)

        self.size = QWidget()
        self.size.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
            QLineEdit {
                background: #222;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit::placeholder {
                color: #888;
            }
        """)
        self.size_layout = QHBoxLayout(self.size)
        self.size_layout.setContentsMargins(0, 5, 0, 0)
        self.xsize = QLineEdit()
        self.xsize.setPlaceholderText("X")
        self.ysize = QLineEdit()
        self.ysize.setPlaceholderText("Y")
        self.zsize = QLineEdit()
        self.zsize.setPlaceholderText("Z")
        self.size_layout.addWidget(self.xsize)
        self.size_layout.addWidget(self.ysize)
        self.size_layout.addWidget(self.zsize)
        menu_layout.addWidget(self.size)

        add_button = QPushButton("Add Object")
        add_button.clicked.connect(lambda: self.add_object(self.add_drop.currentText(), (float(x.text()), float(y.text()), float(z.text())), (float(self.xsize.text()), float(self.ysize.text()), float(self.zsize.text()))))
        menu_layout.addWidget(add_button)
        
        # Disable focus for all widgets
        self.menu_shown = False
    
    def menu_showhide(self):
        self.menu_shown = not self.menu_shown
        if self.menu_shown:
            self.show()
        else:
            self.hide()

    def update_speed(self, value, label):
        self.max_speed = value / 10.0
        label.setText(f"Speed: {self.max_speed:.1f}")
    
    def update_sizes(self):
        match self.add_drop.currentText():
            case "Cube":
                x = QLineEdit()
                x.setPlaceholderText("X")
                y = QLineEdit()
                y.setPlaceholderText("Y")
                z = QLineEdit()
                z.setPlaceholderText("Z")
                self.xsize, self.ysize, self.zsize = x, y, z
            case "Sphere":
                radius = QLineEdit()
                radius.setPlaceholderText("Radius")
                rings = QLineEdit()
                rings.setPlaceholderText("Rings")
                segments = QLineEdit()
                segments.setPlaceholderText("Segments")
                self.xsize, self.ysize, self.zsize = radius, rings, segments
            case "Cone":
                radius = QLineEdit()
                radius.setPlaceholderText("Radius")
                height = QLineEdit()
                height.setPlaceholderText("Height")
                faces = QLineEdit()
                faces.setPlaceholderText("Faces")
                self.xsize, self.ysize, self.zsize = radius, height, faces
    
    def add_object(self, object, position, params):
        match object:
            case "Cube":
                api.resultAPI.create_cube(position, "Cube", params[0], params[1], params[2])
            case "Sphere":
                api.resultAPI.create_sphere(position, "Sphere", params[0], params[1], params[2])
            case "Cone":
                api.resultAPI.create_cone(position, "Cone", params[0], params[1], params[2], False)
        if self.parent():
            self.parent().update()
    
    def reset_camera(self):
        self.cam.Position = numpy.array([0.0, 0.0, 0.0])
        self.cam.yaw = -90.0
        self.cam.pitch = 0.0
        self.velocity = numpy.array([0.0, 0.0, 0.0])