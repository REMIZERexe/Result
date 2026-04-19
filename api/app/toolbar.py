import os
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QLineEdit, QStackedWidget, QColorDialog,
    QSizePolicy, QDialog, QScrollArea, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QSize
import numpy
import api.resultAPI

import math
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# ── Helpers ───────────────────────────────────────────────────────────────────

_STYLE = """
    Toolbar {
        background-color: rgba(30, 30, 30, 220);
        border-right: 2px solid rgba(180, 0, 220, 255);
    }
    QLabel {
        color: #ddd;
        font-size: 13px;
        padding: 2px 6px;
        background: transparent;
    }
    QLabel#section {
        color: white;
        font-size: 15px;
        font-weight: bold;
        letter-spacing: 1px;
        padding: 10px 6px 2px 0;
        background: transparent;
    }
    QPushButton {
        background-color: rgba(120, 0, 140, 255);
        color: white;
        border: none;
        padding: 7px 10px;
        border-radius: 4px;
        font-size: 13px;
    }
    QPushButton:hover  { background-color: rgba(160, 0, 190, 255); }
    QPushButton:pressed{ background-color: rgba(80,  0, 100, 255); }
    QPushButton#flat {
        background: transparent;
        color: #aaa;
        font-size: 11px;
        padding: 4px 6px;
        text-align: left;
    }
    QPushButton#flat:hover { color: white; background: transparent; }
    QSlider::groove:horizontal {
        height: 4px;
        background: rgba(80, 80, 80, 255);
        border-radius: 2px;
        margin: 0 4px;
    }
    QSlider::handle:horizontal {
        background: rgba(160, 0, 200, 255);
        width: 14px; height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QLineEdit {
        background: rgba(20, 20, 20, 255);
        color: white;
        border: 1px solid rgba(100, 0, 120, 255);
        border-radius: 4px;
        padding: 4px 6px;
        font-size: 12px;
    }
    QLineEdit:focus { border-color: rgba(200, 0, 240, 255); }
    QComboBox {
        background: rgba(20, 20, 20, 255);
        color: white;
        border: 1px solid rgba(100, 0, 120, 255);
        border-radius: 4px;
        padding: 4px 6px;
        font-size: 13px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background: rgba(30, 30, 30, 255);
        color: white;
        selection-background-color: rgba(120, 0, 140, 255);
    }
"""

_BROWSER_STYLE = """
    QDialog {
        background-color: rgb(25, 25, 25);
    }
    QScrollArea {
        background-color: rgb(25, 25, 25);
        border: none;
    }
    QWidget#grid_container {
        background-color: rgb(25, 25, 25);
    }
    QLabel {
        color: #ccc;
        font-size: 11px;
        background: transparent;
        padding: 2px 0;
    }
    QPushButton#tex_item,
    _ModelPreview#mdl_item {
        background-color: rgb(40, 40, 40);
        border: 2px solid transparent;
        border-radius: 6px;
        padding: 4px;
    }
    QPushButton#tex_item:hover,
    _ModelPreview#mdl_item:hover {
        border-color: rgba(160, 0, 200, 255);
    }
    QPushButton#tex_item[selected=true],
    _ModelPreview#mdl_item[selected=true] {
        border-color: rgba(220, 0, 255, 255);
        background-color: rgb(60, 0, 80);
    }
    QPushButton#clear_btn {
        background-color: rgb(60, 60, 60);
        color: #aaa;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 12px;
    }
    QPushButton#clear_btn:hover { background-color: rgb(80, 80, 80); color: white; }
    QPushButton#select_btn {
        background-color: rgba(120, 0, 140, 255);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 16px;
        font-size: 13px;
    }
    QPushButton#select_btn:hover { background-color: rgba(160, 0, 190, 255); }
"""

_TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tiff", ".webp"}
_MODEL_EXTS = {".obj", ".gltf", ".glb", ".fbx"}

def _load_preview_geometry(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".obj":             return _prev_obj(path)
        elif ext in (".gltf", ".glb"): return _prev_gltf(path)
        elif ext == ".fbx":           return _prev_fbx(path)
    except Exception as e:
        print(f"[Preview] {path}: {e}")
    return None, None

def _prev_obj(path):
    import re
    raw, verts, tris, cache = [], [], [], {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', line)
                if len(nums) == 3:
                    raw.append([float(n) for n in nums])
            elif line.startswith("f "):
                face = []
                for p in line.split()[1:]:
                    i = int(p.split("/")[0]) - 1
                    if i not in cache:
                        cache[i] = len(verts)
                        verts.append(raw[i])
                    face.append(cache[i])
                for i in range(1, len(face) - 1):
                    tris.extend([face[0], face[i], face[i + 1]])
    if not verts or not tris:
        return None, None
    return numpy.array(verts, dtype=numpy.float32), numpy.array(tris, dtype=numpy.uint32)

def _prev_gltf(path):
    from pygltflib import GLTF2
    import struct, base64
    gltf = GLTF2().load(path)

    def _blob():
        if not gltf.buffers: return b""
        buf = gltf.buffers[0]
        if buf.uri is None: return gltf.binary_blob()
        if buf.uri.startswith("data:"):
            _, enc = buf.uri.split(",", 1)
            return base64.b64decode(enc)
        with open(os.path.join(os.path.dirname(path), buf.uri), "rb") as f:
            return f.read()

    blob = _blob()
    _COMP  = {5120:"b",5121:"B",5122:"h",5123:"H",5125:"I",5126:"f"}
    _COUNT = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

    def _acc(idx):
        acc = gltf.accessors[idx];  bv = gltf.bufferViews[acc.bufferView]
        fmt = _COMP[acc.componentType];  n = _COUNT[acc.type]
        size = struct.calcsize(fmt) * n
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        stride = bv.byteStride or size
        data = []
        for i in range(acc.count):
            chunk = blob[start + i*stride : start + i*stride + size]
            row = list(struct.unpack(f"{n}{fmt}", chunk))
            data.append(row if n > 1 else row[0])
        return data

    all_v, all_t = [], []
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            base = len(all_v)
            pos  = _acc(prim.attributes.POSITION)
            all_v.extend(pos)
            idx  = _acc(prim.indices) if prim.indices is not None else list(range(len(pos)))
            for i in range(0, len(idx) - 2, 3):
                all_t.extend([idx[i]+base, idx[i+1]+base, idx[i+2]+base])
    if not all_v or not all_t:
        return None, None
    return numpy.array(all_v, dtype=numpy.float32), numpy.array(all_t, dtype=numpy.uint32)

def _prev_fbx(path):
    import trimesh
    scene = trimesh.load(path, force="scene")
    meshes = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    mesh = trimesh.util.concatenate(meshes)
    return (numpy.array(mesh.vertices,      dtype=numpy.float32),
            numpy.array(mesh.faces.ravel(), dtype=numpy.uint32))

def _preview_mvp(verts: numpy.ndarray) -> numpy.ndarray:
    mn, mx   = verts.min(axis=0), verts.max(axis=0)
    center   = (mn + mx) / 2.0
    extent   = float((mx - mn).max()) or 1.0
    scale    = 1.9 / extent

    # Center the model
    C        = numpy.eye(4, dtype=numpy.float32)
    C[3, :3] = -center

    # Uniform scale to fit view
    S        = numpy.diag([scale, scale, scale, 1.0]).astype(numpy.float32)

    # Yaw 35° around Y
    yaw = math.radians(35);  cy, sy = math.cos(yaw), math.sin(yaw)
    Ry  = numpy.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=numpy.float32)

    # Pitch -22° (tilt slightly downward)
    pitch = math.radians(-22);  cp, sp = math.cos(pitch), math.sin(pitch)
    Rx    = numpy.array([[1,0,0,0],[0,cp,-sp,0],[0,sp,cp,0],[0,0,0,1]], dtype=numpy.float32)

    # Push model back so camera (at origin) can see it
    Tz       = numpy.eye(4, dtype=numpy.float32)
    Tz[3, 2] = -3.0

    # Perspective projection (same convention as the engine)
    near, far = 0.1, 50.0
    f  = 1.0 / math.tan(math.radians(50) / 2)
    P  = numpy.array([
        [f, 0, 0,  0],
        [0, f, 0,  0],
        [0, 0, far/(near-far), -1],
        [0, 0, (near*far)/(near-far), 0]
    ], dtype=numpy.float32)

    return numpy.ascontiguousarray(C @ S @ Ry @ Rx @ Tz @ P, dtype=numpy.float32)

class _ModelPreview(QOpenGLWidget):
    clicked = pyqtSignal()

    _VERT = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        uniform mat4 MVP;
        flat out vec3 fragNormal;
        void main() {
            gl_Position = vec4(position, 1.0) * MVP;
            fragNormal = normal;
        }
    """
    _FRAG = """
        #version 330 core
        flat in vec3 fragNormal;
        out vec4 FragColor;
        void main() {
            vec3 light = normalize(vec3(1.0, 1.5, 0.8));
            float shade = 0.3 + max(dot(normalize(fragNormal), light), 0.0) * 0.7;
            FragColor = vec4(vec3(0.55, 0.65, 0.92) * shade, 1.0);
        }
    """

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(128, 128)
        self._path        = model_path
        self._shader      = None
        self._vao         = None
        self._index_count = 0
        self._mvp         = None
        self._ready       = False
        self.selected     = False

    def set_selected(self, val: bool):
        self.selected = val
        self.update()

    def mousePressEvent(self, _event):
        self.clicked.emit()

    def initializeGL(self):
        from OpenGL.GL import (glEnable, GL_DEPTH_TEST,
                               glGenVertexArrays, glGenBuffers, glBindVertexArray,
                               glBindBuffer, glBufferData, glEnableVertexAttribArray,
                               glVertexAttribPointer, GL_ARRAY_BUFFER,
                               GL_ELEMENT_ARRAY_BUFFER, GL_FLOAT, GL_FALSE, GL_STATIC_DRAW)
        from OpenGL.GL.shaders import compileShader, compileProgram
        from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER

        glEnable(GL_DEPTH_TEST)

        try:
            self._shader = compileProgram(
                compileShader(self._VERT, GL_VERTEX_SHADER),
                compileShader(self._FRAG, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"[Preview shader] {e}")
            return

        verts, tris = _load_preview_geometry(self._path)
        if verts is None or len(tris) == 0:
            return

        # Re-use the engine's flat-shading expander
        from api.resultAPI import _expand_for_flat_shading
        v4 = numpy.hstack([verts, numpy.ones((len(verts), 1), dtype=numpy.float32)])
        exp_v, exp_n, _, seq_idx = _expand_for_flat_shading(v4, tris)

        self._mvp         = _preview_mvp(exp_v)
        self._index_count = len(seq_idx)

        self._vao          = glGenVertexArrays(1)
        vbo_pos, vbo_nrm, ebo = glGenBuffers(3)

        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, exp_v.nbytes, exp_v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_nrm)
        glBufferData(GL_ARRAY_BUFFER, exp_n.nbytes, exp_n, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, seq_idx.nbytes, seq_idx, GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._ready = True

    def paintGL(self):
        from OpenGL.GL import (glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                               glUseProgram, glGetUniformLocation, glUniformMatrix4fv, GL_TRUE,
                               glBindVertexArray, glDrawElements, GL_TRIANGLES, GL_UNSIGNED_INT)

        # Purple tint when selected, dark grey otherwise
        if self.selected:
            glClearColor(0.28, 0.0, 0.38, 1.0)
        else:
            glClearColor(0.13, 0.13, 0.13, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self._ready:
            return

        glUseProgram(self._shader)
        glUniformMatrix4fv(glGetUniformLocation(self._shader, "MVP"),
                           1, GL_TRUE, self._mvp)
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section")
    return lbl

def _labeled_field(placeholder: str, default: str,
                   validator=None) -> QLineEdit:
    field = QLineEdit()
    field.setPlaceholderText(placeholder)
    field.setText(default)
    if validator:
        field.setValidator(validator)
    return field

def _row(*widgets) -> QWidget:
    container = QWidget()
    container.setObjectName("row")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(6)
    for w in widgets:
        layout.addWidget(w)
    return container

# ── Browsers ───────────────────────────────────────────────────────────

class TextureBrowser(QDialog):
    def __init__(self, current_texture: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Texture Browser")
        self.setMinimumSize(520, 420)
        self.setStyleSheet(_BROWSER_STYLE)
        self.selected_texture = current_texture  # None = no texture

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Scroll area with grid of thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        self._grid_container = QWidget()
        self._grid_container.setObjectName("grid_container")
        self._grid = QGridLayout(self._grid_container)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(self._grid_container)

        # Bottom bar: selected label + buttons
        bottom = QHBoxLayout()
        self._selected_label = QLabel("None selected")
        self._selected_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 0;")
        bottom.addWidget(self._selected_label)
        bottom.addStretch()

        clear_btn = QPushButton("No Texture")
        clear_btn.setObjectName("clear_btn")
        clear_btn.clicked.connect(self._clear_texture)
        bottom.addWidget(clear_btn)

        select_btn = QPushButton("Select")
        select_btn.setObjectName("select_btn")
        select_btn.clicked.connect(self.accept)
        bottom.addWidget(select_btn)

        root.addLayout(bottom)

        self._item_buttons = {}
        self._populate()

        if current_texture:
            self._highlight(current_texture)

    def _populate(self):
        assets_path = os.path.join(
            api.resultAPI.get_assets_path(), "assets", "textures"
        )
        if not os.path.isdir(assets_path):
            lbl = QLabel(f"Folder not found:\n{assets_path}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lbl, 0, 0)
            return

        files = sorted(
            f for f in os.listdir(assets_path)
            if os.path.splitext(f)[1].lower() in _TEXTURE_EXTS
        )

        if not files:
            lbl = QLabel("No textures found in assets/textures/")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lbl, 0, 0)
            return

        cols = 3
        for idx, filename in enumerate(files):
            tex_name = os.path.splitext(filename)[0]
            full_path = os.path.join(assets_path, filename)

            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(3)
            cell_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            # Thumbnail button
            btn = QPushButton()
            btn.setObjectName("tex_item")
            btn.setFixedSize(128, 128)
            btn.setIconSize(QSize(116, 116))

            pixmap = QPixmap(full_path)
            if not pixmap.isNull():
                btn.setIcon(QIcon(pixmap.scaled(
                    116, 116,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )))

            btn.clicked.connect(lambda checked, n=tex_name: self._on_select(n))
            btn.setProperty("selected", "false")
            self._item_buttons[tex_name] = btn
            cell_layout.addWidget(btn)

            # Filename label below thumbnail
            name_lbl = QLabel(tex_name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            name_lbl.setWordWrap(True)
            cell_layout.addWidget(name_lbl)

            self._grid.addWidget(cell, idx // cols, idx % cols)

    def _on_select(self, tex_name: str):
        self._highlight(tex_name)

    def _highlight(self, tex_name: str):
        # Deselect all
        for name, btn in self._item_buttons.items():
            btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Select clicked one
        if tex_name in self._item_buttons:
            self._item_buttons[tex_name].setProperty("selected", "true")
            self._item_buttons[tex_name].style().unpolish(self._item_buttons[tex_name])
            self._item_buttons[tex_name].style().polish(self._item_buttons[tex_name])

        self.selected_texture = tex_name
        self._selected_label.setText(f"Selected: {tex_name}")

    def _clear_texture(self):
        for btn in self._item_buttons.values():
            btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.selected_texture = None
        self._selected_label.setText("None selected")

class ModelBrowser(QDialog):
    def __init__(self, current_model: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Browser")
        self.setMinimumSize(520, 480)
        self.setStyleSheet(_BROWSER_STYLE)
        self.selected_model      = current_model
        self.selected_model_name = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        self._grid_container = QWidget()
        self._grid_container.setObjectName("grid_container")
        self._grid = QGridLayout(self._grid_container)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(self._grid_container)

        bottom = QHBoxLayout()
        self._selected_label = QLabel("None selected")
        self._selected_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 0;")
        bottom.addWidget(self._selected_label)
        bottom.addStretch()

        clear_btn = QPushButton("No Model")
        clear_btn.setObjectName("clear_btn")
        clear_btn.clicked.connect(self._clear_model)
        bottom.addWidget(clear_btn)

        select_btn = QPushButton("Select")
        select_btn.setObjectName("select_btn")
        select_btn.clicked.connect(self.accept)
        bottom.addWidget(select_btn)

        root.addLayout(bottom)

        self._item_previews = {}
        self._populate()

        if current_model:
            self._highlight(current_model)

    def _populate(self):
        _MODEL_EXTS = {".obj", ".gltf", ".glb", ".fbx"}
        assets_path = os.path.join(api.resultAPI.get_assets_path(), "assets", "models")

        if not os.path.isdir(assets_path):
            self._grid.addWidget(QLabel(f"Folder not found:\n{assets_path}"), 0, 0)
            return

        files = sorted(f for f in os.listdir(assets_path)
                       if os.path.splitext(f)[1].lower() in _MODEL_EXTS)

        if not files:
            self._grid.addWidget(QLabel("No models found in assets/models/"), 0, 0)
            return

        cols = 3
        for idx, filename in enumerate(files):
            mdl_name  = os.path.splitext(filename)[0]
            full_path = os.path.join(assets_path, filename)

            cell        = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(3)
            cell_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            preview = _ModelPreview(full_path)
            preview.setObjectName("mdl_item")
            preview.clicked.connect(
                lambda n=mdl_name, p=full_path: self._on_select(n, p)
            )
            self._item_previews[mdl_name] = preview
            cell_layout.addWidget(preview)

            name_lbl = QLabel(mdl_name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            name_lbl.setWordWrap(True)
            cell_layout.addWidget(name_lbl)

            self._grid.addWidget(cell, idx // cols, idx % cols)

    def _on_select(self, mdl_name: str, path: str):
        self._highlight(mdl_name)
        self.selected_model      = path
        self.selected_model_name = mdl_name

    def _highlight(self, mdl_name: str):
        for name, prev in self._item_previews.items():
            prev.set_selected(name == mdl_name)
        self._selected_label.setText(f"Selected: {mdl_name}")

    def _clear_model(self):
        for prev in self._item_previews.values():
            prev.set_selected(False)
        self.selected_model      = None
        self.selected_model_name = None
        self._selected_label.setText("None selected")

# ── Per-object parameter pages ────────────────────────────────────────────────

class _CubePage(QWidget):
    def __init__(self):
        super().__init__()
        v = QDoubleValidator(0.01, 1e6, 4)
        self.width  = _labeled_field("Width",  "20", v)
        self.depth  = _labeled_field("Depth",  "20", v)
        self.height = _labeled_field("Height", "20", v)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_row(self.width, self.depth, self.height))

    def params(self):
        return (float(self.width.text()  or 20),
                float(self.depth.text()  or 20),
                float(self.height.text() or 20))

class _SpherePage(QWidget):
    def __init__(self):
        super().__init__()
        self.radius   = _labeled_field("Radius",   "20", QDoubleValidator(0.01, 1e6, 4))
        self.segments = _labeled_field("Segments", "32", QIntValidator(3, 256))
        self.rings    = _labeled_field("Rings",    "16", QIntValidator(2, 256))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_row(self.radius, self.segments, self.rings))

    def params(self):
        return (float(self.radius.text()   or 20),
                int(self.segments.text()   or 32),
                int(self.rings.text()      or 16))

class _ConePage(QWidget):
    def __init__(self):
        super().__init__()
        self.radius = _labeled_field("Radius", "20")
        self.height = _labeled_field("Height", "20")
        self.faces  = _labeled_field("Faces",  "15")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_row(self.radius, self.height, self.faces))

    def params(self):
        return (float(self.radius.text() or 20),
                float(self.height.text() or 20),
                int(self.faces.text()    or 15))

class _ModelPage(QWidget):
    def __init__(self):
        super().__init__()
        self.scalex = _labeled_field("Scale X", "1.0")
        self.scaley = _labeled_field("Scale Y", "1.0")
        self.scalez = _labeled_field("Scale Z", "1.0")
        self._model_btn = QPushButton("Model: None")
        self._model_btn.clicked.connect(self._pick_model)
        self._selected_model = None
        self._selected_model_name = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_row(self.scalex, self.scaley, self.scalez))
        layout.addWidget(self._model_btn)

    def params(self):
        return (float(self.scalex.text() or 20),
                float(self.scaley.text() or 20),
                float(self.scalez.text()   or 15))
    
    def _pick_model(self):
        browser = ModelBrowser(self._selected_model, parent=self)
        if browser.exec():
            self._selected_model = browser.selected_model
            self._selected_model_name = browser.selected_model_name
            self._refresh_model_btn()

    def _refresh_model_btn(self):
        if self._selected_model_name:
            self._model_btn.setText(f"Model: {self._selected_model_name}")
            self._model_btn.setStyleSheet(
                "background-color: rgba(60, 0, 80, 255);"
                "color: white; border: 1px solid rgba(200, 0, 240, 255);"
                "border-radius: 4px; padding: 7px 10px;"
            )
        else:
            self._model_btn.setText("Model: none")
            self._model_btn.setStyleSheet("")

# ── Main toolbar ──────────────────────────────────────────────────────────────

_OBJECT_TYPES   = ["Cube", "Sphere", "Cone", "Custom Model"]
_object_counter = {}

def _unique_name(kind: str) -> str:
    _object_counter[kind] = _object_counter.get(kind, 0) + 1
    return f"{kind}_{_object_counter[kind]}"

class Toolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Toolbar")
        self.setFixedWidth(240)
        self.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        cfg = api.resultAPI.config
        max_coord = cfg.getfloat("settings", "max_world_coordinate")

        self._selected_color   = (0.0, 0.5, 1.0, 1.0)
        self._selected_texture = None

        self.setStyleSheet(_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 12, 10, 12)
        root.setSpacing(2)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("Result3D")
        title.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: white;"
            "padding: 4px 10px 8px 10px; background: transparent;"
        )
        root.addWidget(title)

        # ── Camera section ─────────────────────────────────────────────────
        root.addWidget(_section_label("Camera"))

        max_speed = cfg.getfloat("camera", "max_speed")
        self._speed_label = QLabel(f"Speed: {max_speed:.1f}")
        root.addWidget(self._speed_label)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setMinimum(1)
        self._speed_slider.setMaximum(100)
        self._speed_slider.setValue(int(max_speed * 10))
        self._speed_slider.setContentsMargins(6, 0, 6, 0)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        root.addWidget(self._speed_slider)

        reset_btn = QPushButton("Reset Camera")
        reset_btn.setObjectName("flat")
        reset_btn.clicked.connect(self._reset_camera)
        root.addWidget(reset_btn)

        # ── Objects section ────────────────────────────────────────────────
        root.addWidget(_section_label("Create Object"))

        self._type_combo = QComboBox()
        self._type_combo.addItems(_OBJECT_TYPES)
        self._type_combo.setContentsMargins(6, 0, 6, 0)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        root.addWidget(self._type_combo)

        # Position row
        coord_v = QDoubleValidator(-max_coord, max_coord, 4)
        self._pos_x = _labeled_field("X", "0", coord_v)
        self._pos_y = _labeled_field("Y", "0", coord_v)
        self._pos_z = _labeled_field("Z", "0", coord_v)
        root.addWidget(_row(self._pos_x, self._pos_y, self._pos_z))

        # Stacked param pages
        self._pages       = QStackedWidget()
        self._page_cube   = _CubePage()
        self._page_sphere = _SpherePage()
        self._page_cone   = _ConePage()
        self._page_model  = _ModelPage()
        self._pages.addWidget(self._page_cube)
        self._pages.addWidget(self._page_sphere)
        self._pages.addWidget(self._page_cone)
        self._pages.addWidget(self._page_model)
        root.addWidget(self._pages)

        # Color picker
        self._color_btn = QPushButton("Color")
        self._color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        root.addWidget(self._color_btn)

        # Texture picker
        self._texture_btn = QPushButton("Texture: None")
        self._texture_btn.clicked.connect(self._pick_texture)
        root.addWidget(self._texture_btn)

        # Add button
        add_btn = QPushButton("Add Object")
        add_btn.clicked.connect(self._add_object)
        root.addWidget(add_btn)

        # ── Rendering ──────────────────────────────────────────────────────
        root.addWidget(_section_label("Rendering"))

        self.solid     = True
        self.textured  = False
        self.lit       = False
        self.wireframe = True

        solid_btn = QPushButton()
        solid_btn.setCheckable(True)
        solid_btn.setIcon(QIcon("assets/icons/solid.png"))
        solid_btn.toggled.connect(self._solid_toggle)

        textured_btn = QPushButton()
        textured_btn.setCheckable(True)
        textured_btn.setIcon(QIcon("assets/icons/textured.png"))
        textured_btn.toggled.connect(self._textured_toggle)

        lit_btn = QPushButton()
        lit_btn.setCheckable(True)
        lit_btn.setIcon(QIcon("assets/icons/lit.png"))
        lit_btn.toggled.connect(self._lit_toggle)
        root.addWidget(_row(solid_btn, textured_btn, lit_btn))

        wireframe_btn = QPushButton()
        wireframe_btn.setCheckable(True)
        wireframe_btn.setIcon(QIcon("assets/icons/wireframe.png"))
        wireframe_btn.toggled.connect(self._wireframe_toggle)
        root.addWidget(wireframe_btn)

        root.addStretch(1)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _solid_toggle(self):
        self.solid = not self.solid
        if self.solid:
            self.textured = False

    def _textured_toggle(self):
        self.textured = not self.textured
        if self.textured:
            self.solid = False

    def _lit_toggle(self):
        self.lit = not self.lit

    def _wireframe_toggle(self):
        self.wireframe = not self.wireframe

    def _on_speed_changed(self, value: int):
        speed = value / 10.0
        self._speed_label.setText(f"Speed: {speed:.1f}")
        if self.parent():
            self.parent().max_speed = speed

    def _on_type_changed(self, index: int):
        self._pages.setCurrentIndex(index)

    def _pick_color(self):
        r, g, b, _ = self._selected_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        col = QColorDialog.getColor(initial, self, "Object Color")
        if col.isValid():
            self._selected_color = (col.redF(), col.greenF(), col.blueF(), 1.0)
            self._refresh_color_btn()

    def _refresh_color_btn(self):
        r, g, b, _ = self._selected_color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});"
            "color: white; border-radius: 4px; padding: 7px 10px;"
        )

    def _pick_texture(self):
        browser = TextureBrowser(self._selected_texture, parent=self)
        if browser.exec():
            self._selected_texture = browser.selected_texture
            self._auto_load_texture(self._selected_texture)
            self._refresh_texture_btn()
    
    def _auto_load_texture(self, tex_name: str | None):
        if tex_name is None:
            return
        # Skip if already loaded or already pending
        if (tex_name in api.resultAPI.result.Textures or
            tex_name in api.resultAPI.result.PendingTextures):
            return
        # Find the file in assets/textures/
        assets_path = os.path.join(api.resultAPI.get_assets_path(), "assets", "textures")
        for filename in os.listdir(assets_path):
            name, ext = os.path.splitext(filename)
            if name == tex_name and ext.lower() in _TEXTURE_EXTS:
                full_path = os.path.join(assets_path, filename)
                api.resultAPI.load_texture(tex_name, full_path)
                print(f"[Toolbar] Auto-loaded texture: {tex_name}")
                return
        print(f"[Toolbar] Warning: texture file for '{tex_name}' not found")

    def _refresh_texture_btn(self):
        if self._selected_texture:
            self._texture_btn.setText(f"Texture: {self._selected_texture}")
            self._texture_btn.setStyleSheet(
                "background-color: rgba(60, 0, 80, 255);"
                "color: white; border: 1px solid rgba(200, 0, 240, 255);"
                "border-radius: 4px; padding: 7px 10px;"
            )
        else:
            self._texture_btn.setText("Texture: none")
            self._texture_btn.setStyleSheet("")

    def _add_object(self):
        try:
            pos = (
                float(self._pos_x.text() or 0),
                float(self._pos_y.text() or 0),
                float(self._pos_z.text() or 0),
            )
            kind  = _OBJECT_TYPES[self._type_combo.currentIndex()]
            name  = _unique_name(kind)
            color = self._selected_color
            page  = self._pages.currentWidget()

            match kind:
                case "Cube":
                    w, d, h = page.params()
                    api.resultAPI.create_cube(pos, name, w, d, h, color=color)
                case "Sphere":
                    r, segs, rings = page.params()
                    api.resultAPI.create_sphere(pos, name, r, segs, rings, color=color)
                case "Cone":
                    r, h, faces = page.params()
                    api.resultAPI.create_cone(pos, name, r, h, faces, False, color=color)
                case "Custom Model":
                    sx, sy, sz = page.params()
                    api.resultAPI.load_model(self._page_model._selected_model, pos, name, color=color, scale_x=sx, scale_y=sy, scale_z=sz)

            # Apply selected texture immediately after creation
            if self._selected_texture:
                api.resultAPI.set_object_texture(name, self._selected_texture)

            if self.parent():
                self.parent().update()

        except ValueError as e:
            print(f"[Toolbar] Invalid input: {e}")

    def _reset_camera(self):
        if self.parent():
            win = self.parent()
            win.cam.Position = numpy.array([0.0, 0.0, 0.0])
            win.cam.Yaw   = -90.0
            win.cam.Pitch =   0.0
            win.velocity  = numpy.array([0.0, 0.0, 0.0])

    # ── Public API ─────────────────────────────────────────────────────────

    def toggle(self):
        self.setVisible(not self.isVisible())