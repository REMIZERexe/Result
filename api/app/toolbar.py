from PyQt6.QtGui import QDoubleValidator, QIntValidator, QColor, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QLineEdit, QStackedWidget, QColorDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
import numpy
import api.resultAPI

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
    """Pack widgets side-by-side with no extra margins."""
    container = QWidget()
    container.setObjectName("row")         # won't match any broad selector
    layout = QHBoxLayout(container)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(6)
    for w in widgets:
        layout.addWidget(w)
    return container

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
        return (float(self.radius.text()      or 20),
                int(self.segments.text()  or 32),
                int(self.rings.text()     or 16))

class _ConePage(QWidget):
    def __init__(self):
        super().__init__()
        self.radius = _labeled_field("Radius", "20", QDoubleValidator(0.01, 1e6, 4))
        self.height = _labeled_field("Height", "20", QDoubleValidator(0.01, 1e6, 4))
        self.faces  = _labeled_field("Faces",  "15", QIntValidator(3, 256))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_row(self.radius, self.height, self.faces))

    def params(self):
        return (float(self.radius.text() or 20),
                float(self.height.text() or 20),
                int(self.faces.text()    or 15))

# ── Main toolbar ──────────────────────────────────────────────────────────────

# Maps combo index → (page index, display name)
_OBJECT_TYPES = ["Cube", "Sphere", "Cone"]
_object_counter = {}   # {"Cube": 0, "Sphere": 2, ...}  for unique naming

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

        self._selected_color = (0.0, 0.5, 1.0, 1.0)

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

        # Stacked param pages (one per object type, order matches _OBJECT_TYPES)
        self._pages = QStackedWidget()
        self._page_cube   = _CubePage()
        self._page_sphere = _SpherePage()
        self._page_cone   = _ConePage()
        self._pages.addWidget(self._page_cube)
        self._pages.addWidget(self._page_sphere)
        self._pages.addWidget(self._page_cone)
        root.addWidget(self._pages)

        # Color picker
        self._color_btn = QPushButton("Color")
        self._color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        root.addWidget(self._color_btn)

        # Add button
        add_btn = QPushButton("Add Object")
        add_btn.clicked.connect(self._add_object)
        root.addWidget(add_btn)

        # ── Rendering ──────────────────────────────────────────────────────
        root.addWidget(_section_label("Rendering"))
        
        # Initial states
        self.solid = True
        self.textured = False
        self.lit = False
        self.wireframe = True

        # Solid-rendering button
        solid_btn = QPushButton()
        solid_btn.setCheckable(True)
        solid_btn.setIcon(QIcon("assets/icons/solid.png"))
        solid_btn.toggled.connect(self._solid_toggle)

        # Textured button
        textured_btn = QPushButton()
        textured_btn.setCheckable(True)
        textured_btn.setIcon(QIcon("assets/icons/textured.png"))
        textured_btn.toggled.connect(self._textured_toggle)

        # Lit button
        lit_btn = QPushButton()
        lit_btn.setCheckable(True)
        lit_btn.setIcon(QIcon("assets/icons/lit.png"))
        lit_btn.toggled.connect(self._lit_toggle)
        root.addWidget(_row(solid_btn, textured_btn, lit_btn))

        # Wireframe button
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
        # Propagate to the engine window that owns physics state
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

            if self.parent():
                self.parent().update()

        except ValueError as e:
            print(f"[Toolbar] Invalid input: {e}")
    
    def _set_solid(self):
        self.solid = True
    def _set_textured(self):
        self.textured = True
    def _set_lit(self):
        self.lit = True
    def _toggle_wireframe(self):
        self.wireframe = True
    
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