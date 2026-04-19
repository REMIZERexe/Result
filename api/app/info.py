import sys

sys.dont_write_bytecode = True
import numpy
import time

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QCursor
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QMenu, QComboBox, QLineEdit

from api.app.toolbar import Toolbar

class Info(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setObjectName("Info")
        self.setFixedWidth(240)

        self.setStyleSheet('''
            QLabel {
                color: rgb(255, 255, 150)
            }
        ''')
        root = QVBoxLayout(self)
        root.setSpacing(1)
        root.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.fps = QLabel("")
        root.addWidget(self.fps)

        root.addStretch()