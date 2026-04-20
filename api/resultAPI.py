import sys

sys.dont_write_bytecode = True
from configparser import ConfigParser
config = ConfigParser()
import os

def get_base_path():
    import sys
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    # api/resultAPI.py → dirname → api/ → join app/ → api/app/
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")

config.read(os.path.join(get_base_path(), "config.ini"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Callable
import math
import numpy
import random
from opensimplex import OpenSimplex
from OpenGL.GL import (GL_NEAREST, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR)

def get_assets_path():
    import sys
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    # Go up from api/resultAPI.py two levels to project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _triangulate_quads(faces) -> list:
    """Convert quad face list [(a,b,c,d), ...] into triangle index list."""
    tris = []
    for f in faces:
        if len(f) == 3:
            tris.extend(f)
        elif len(f) == 4:
            a, b, c, d = f
            tris.extend([a, b, c, a, c, d])
        else:  # polygon fan
            for i in range(1, len(f) - 1):
                tris.extend([f[0], f[i], f[i + 1]])
    return tris

def _compute_smooth_normals(vertices, triangles):
    """Average normals across shared vertices for smooth shading."""
    verts = numpy.array(vertices, dtype=numpy.float32)
    if verts.ndim == 2 and verts.shape[1] == 4:
        verts = verts[:, :3]
    tris  = numpy.array(triangles, dtype=numpy.int32).reshape(-1, 3)

    normals = numpy.zeros_like(verts)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_normals = numpy.cross(v1 - v0, v2 - v0)

    # Accumulate face normals into each vertex
    for i in range(3):
        numpy.add.at(normals, tris[:, i], face_normals)

    # Normalize
    lengths = numpy.linalg.norm(normals, axis=1, keepdims=True)
    lengths = numpy.where(lengths == 0.0, 1.0, lengths)
    normals /= lengths

    seq_indices = numpy.arange(len(verts), dtype=numpy.uint32)
    return (verts.astype(numpy.float32),
            normals.astype(numpy.float32),
            seq_indices)

def _compute_flat_normals(vertices, triangles) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Returns (expanded_verts, expanded_normals, sequential_indices) where
    every triangle has its own 3 vertices so normals are never averaged.
    Use these instead of the original verts/tris when uploading to GPU.
    """
    verts = numpy.array(vertices, dtype=numpy.float32)
    if verts.ndim == 2 and verts.shape[1] == 4:
        verts = verts[:, :3]

    tris = numpy.array(triangles, dtype=numpy.int32).reshape(-1, 3)
    n_tris = len(tris)

    # Expand: each triangle gets 3 unique vertices
    expanded_verts   = verts[tris.ravel()].reshape(n_tris * 3, 3)   # (N*3, 3)

    # Per-face normal, same for all 3 vertices of that face
    v0 = expanded_verts[0::3]
    v1 = expanded_verts[1::3]
    v2 = expanded_verts[2::3]
    face_normals = numpy.cross(v1 - v0, v2 - v0)                    # (N, 3)

    lengths = numpy.linalg.norm(face_normals, axis=1, keepdims=True)
    lengths = numpy.where(lengths == 0.0, 1.0, lengths)
    face_normals /= lengths

    # Repeat each face normal 3 times
    expanded_normals = numpy.repeat(face_normals, 3, axis=0)         # (N*3, 3)

    # Sequential indices — 0,1,2, 3,4,5, ...
    sequential_indices = numpy.arange(n_tris * 3, dtype=numpy.uint32)

    return (expanded_verts.astype(numpy.float32),
            expanded_normals.astype(numpy.float32),
            sequential_indices)

def _expand_for_flat_shading(vertices, triangles, uvs=None):
    """
    Duplicates vertices so every triangle is independent (required for flat normals).
    Also expands UVs the same way if provided.
    Returns (exp_verts, exp_normals, exp_uvs, seq_indices).
    """
    verts = numpy.array(vertices, dtype=numpy.float32)
    if verts.ndim == 2 and verts.shape[1] == 4:
        verts = verts[:, :3]

    tris   = numpy.array(triangles, dtype=numpy.int32).reshape(-1, 3)
    n_tris = len(tris)

    exp_verts = verts[tris.ravel()].reshape(n_tris * 3, 3)

    v0 = exp_verts[0::3]
    v1 = exp_verts[1::3]
    v2 = exp_verts[2::3]
    face_normals = numpy.cross(v1 - v0, v2 - v0)
    lengths      = numpy.linalg.norm(face_normals, axis=1, keepdims=True)
    lengths      = numpy.where(lengths == 0.0, 1.0, lengths)
    face_normals /= lengths
    exp_normals   = numpy.repeat(face_normals, 3, axis=0)

    if uvs is not None:
        uv_arr  = numpy.array(uvs, dtype=numpy.float32).reshape(-1, 2)
        exp_uvs = uv_arr[tris.ravel()].reshape(n_tris * 3, 2)
    else:
        exp_uvs = numpy.zeros((n_tris * 3, 2), dtype=numpy.float32)

    seq_indices = numpy.arange(n_tris * 3, dtype=numpy.uint32)

    return (exp_verts.astype(numpy.float32),
            exp_normals.astype(numpy.float32),
            exp_uvs.astype(numpy.float32),
            seq_indices)

def _normalize_color(color):
    if max(color[:3]) > 1.0:
        return (color[0]/255, color[1]/255, color[2]/255, color[3])
    return color

def _timer(func: Callable) -> None:
    import time
    def wrapper(*args, **kwargs) -> float:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(end - start) + " seconds")
        return result
    return wrapper

_FILTERS = {
    "nearest": (GL_NEAREST, GL_NEAREST),
    "linear":  (GL_LINEAR,  GL_LINEAR_MIPMAP_LINEAR),
    "bicubic": (GL_LINEAR,  GL_LINEAR_MIPMAP_LINEAR)
}

#! Result classes
class Model:
    """All geometry and material data for a 3D object."""
    def __init__(self):
        self.vertices    = []
        self.edges       = []
        self.triangles   = []
        self.uvs         = []
        self.flat_shading = True
        self.texture_name = None
        self.tiling      = (1.0, 1.0)
        self.tex_filter  = "linear"

class SceneObject:
    """An entity placed in the scene with position, rotation, and a Model."""
    def __init__(self, name="", position=(0, 0, 0), rotation=(0.0, 0.0, 0.0), color=(1, 1, 1, 1), model=Model()):
        self.name     = name
        self.position = numpy.array([position[0], position[1], position[2], 1.0], dtype=numpy.float32)
        self.rotation = (0.0, 0.0, 0.0)   # degrees: (rx, ry, rz)
        self.color    = list(color)
        self.model    = Model()

class Camera(SceneObject):
    def __init__(self) -> None:
        super().__init__()
        self.name = "MainCam"

        self.Yaw   = 0
        self.Pitch = 0
        self.Roll  = 0
        self.Fov   = 110

class DirectionalLight:
    def __init__(self):
        self.direction       = [0.5, -1.0, 0.3]   # world-space direction the light points
        self.color           = [1.0, 1.0, 1.0]
        self.ambient_strength = 0.2

class Scene:
    def __init__(self) -> None:
        self.ObjectsOnScene = []
        self.MainCamera = None

class Matrices:
    def __init__(self) -> None:
        self.aspect_ratio = config.getint("window", "width") / config.getint("window", "height")
        self.near = 0.1
        self.far = 1500.0
        self.f = 1 / math.tan(math.radians(result.MainScene.MainCamera.Fov) / 2)

        self.ProjMatrix = numpy.array([
            [self.f / self.aspect_ratio, 0, 0, 0],
            [0, self.f, 0, 0],
            [0, 0, self.far / (self.near - self.far), -1],
            [0, 0, (self.near * self.far) / (self.near - self.far), 0]
        ])

    def getxRot_matrix(self, angle) -> numpy.ndarray:
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)
        matrix = numpy.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
        return matrix
    
    def getyRot_matrix(self, angle) -> numpy.ndarray:
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)
        matrix = numpy.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        return matrix
    
    def getzRot_matrix(self, angle) -> numpy.ndarray:
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)
        matrix = numpy.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return matrix
    
    def getTrans_matrix(self, tx, ty, tz) -> numpy.ndarray:
        translation_matrix = numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1]
        ])
        return translation_matrix
    
    def getView_matrix(self) -> numpy.ndarray:
        cam = result.MainScene.MainCamera
        cameraPos = cam.Position[:3]  # Make sure it's 3D
        
        # Calculate forward direction from yaw and pitch
        yaw_rad = numpy.radians(cam.Yaw)
        pitch_rad = numpy.radians(cam.Pitch)
        
        # Forward vector (where camera is looking)
        forward = numpy.array([
            numpy.cos(pitch_rad) * numpy.cos(yaw_rad),
            numpy.sin(pitch_rad),
            numpy.cos(pitch_rad) * numpy.sin(yaw_rad)
        ])
        forward = forward / numpy.linalg.norm(forward)
        
        # World up is ALWAYS up in world space
        worldUp = numpy.array([0.0, 1.0, 0.0])
        
        # Right vector (perpendicular to forward and world up)
        right = numpy.cross(forward, worldUp)
        right = right / numpy.linalg.norm(right)
        
        # Camera's up vector (perpendicular to right and forward)
        up = numpy.cross(right, forward)
        # up is already normalized
        
        # Build view matrix
        view_matrix = numpy.array([
            [right[0], up[0], -forward[0], 0],
            [right[1], up[1], -forward[1], 0],
            [right[2], up[2], -forward[2], 0],
            [-numpy.dot(right, cameraPos), -numpy.dot(up, cameraPos), numpy.dot(forward, cameraPos), 1]
        ])
        
        return view_matrix
    
class Result:
    def __init__(self) -> None:
        self.MainScene = None
        self.Matrices = None
        self.WindowParam = None

        # Rendering Settings
        self.solid = False
        self.textured = True
        self.lit = False
        self.wireframe = False

        self.VertexBuffer = []
        self.EdgeBuffer = []
        self.GPUBuffers = {}
        self.RenderList = []
        self.Textures = {}
        self.PendingTextures = {}
        self.Lights = []
#! ----------

#! Settings by default
def set_result_instance() -> None:
    global result
    result = Result()

def set_window_instance(win) -> None:
    global engine_window
    engine_window = win

def set_scene_instance(scene) -> None:
    result.MainScene = scene

def set_camera_instance(camera) -> None:
    result.MainScene.MainCamera = camera

def set_matrices_instance(instance) -> None:
    result.Matrices = instance

def create_directional_light(direction=(0.5, -1.0, 0.3),
                              color=(1.0, 1.0, 1.0),
                              ambient: float = 0.2) -> "DirectionalLight":
    light                  = DirectionalLight()
    light.direction        = list(direction)
    light.color            = list(color)
    light.ambient_strength = ambient
    result.Lights.append(light)
    return light          # caller can hold a reference to update it later

def remove_light(light: "DirectionalLight") -> None:
    if light in result.Lights:
        result.Lights.remove(light)

def clear_lights() -> None:
    result.Lights.clear()

def set_light_direction(light: "DirectionalLight", direction: tuple) -> None:
    light.direction = list(direction)

def set_light_color(light: "DirectionalLight", color: tuple) -> None:
    light.color = list(color)

def set_light_ambient(light: "DirectionalLight", ambient: float) -> None:
    light.ambient_strength = ambient
#! ----------

#! Rendering functions
def render_scene() -> None:
    result.RenderList.clear()
    VIEW = result.Matrices.getView_matrix()
    PROJ = result.Matrices.ProjMatrix

    for obj in result.MainScene.ObjectsOnScene:
        TRANS = result.Matrices.getTrans_matrix(*obj.position[:3])
        rx, ry, rz = obj.rotation
        ROT = (result.Matrices.getxRot_matrix(rx)
             @ result.Matrices.getyRot_matrix(ry)
             @ result.Matrices.getzRot_matrix(rz))
        MVP = numpy.ascontiguousarray((ROT @ TRANS @ VIEW @ PROJ), dtype=numpy.float32)
        result.RenderList.append((obj.name, MVP, obj.color, obj.model.flat_shading))

def load_texture(texture_name: str, path: str, tex_filter: str = "linear") -> None:
    from PIL import Image
    try:
        img  = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        data = numpy.array(img, dtype=numpy.uint8)
        result.PendingTextures[texture_name] = (data, img.width, img.height, tex_filter)
    except Exception as e:
        print(f"load_texture({texture_name}): {e}")

def _upload_texture_to_gpu(texture_name: str, data, width: int, height: int,
                            tex_filter: str = "linear") -> None:
    from OpenGL.GL import (glGenTextures, glBindTexture, glTexImage2D,
                           glTexParameteri, glGenerateMipmap,
                           GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE,
                           GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
                           GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_REPEAT)

    mag, min_ = _FILTERS.get(tex_filter, (GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR))

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)

    result.Textures[texture_name] = tex_id

def upload_object_to_gpu(obj: SceneObject) -> None:
    from OpenGL.GL import (glGenVertexArrays, glGenBuffers, glBindVertexArray,
                           glBindBuffer, glBufferData, glEnableVertexAttribArray,
                           glVertexAttribPointer, GL_ARRAY_BUFFER,
                           GL_ELEMENT_ARRAY_BUFFER, GL_FLOAT, GL_FALSE, GL_STATIC_DRAW)

    mdl  = obj.model
    name = obj.name

    raw_verts = numpy.array(mdl.vertices, dtype=numpy.float32)
    if raw_verts.ndim == 2 and raw_verts.shape[1] == 4:
        raw_verts = raw_verts[:, :3]

    raw_edges = numpy.array(mdl.edges, dtype=numpy.uint32).ravel() if mdl.edges else numpy.array([], dtype=numpy.uint32)
    raw_tris  = numpy.array(mdl.triangles, dtype=numpy.uint32).ravel() if mdl.triangles else numpy.array([], dtype=numpy.uint32)

    raw_mat_groups = mdl.texture_name if isinstance(mdl.texture_name, list) else None

    if mdl.uvs and not raw_mat_groups:
        tx, ty = mdl.tiling
        scaled_uvs = [[u * tx, v * ty] for u, v in mdl.uvs]
    else:
        scaled_uvs = mdl.uvs or []

    if len(raw_tris) > 0:
        if mdl.flat_shading:
            exp_verts, exp_normals, exp_uvs, seq_indices = \
                _expand_for_flat_shading(mdl.vertices, mdl.triangles, scaled_uvs)
        else:
            exp_verts, exp_normals, _ = _compute_smooth_normals(mdl.vertices, mdl.triangles)
            seq_indices = raw_tris
            exp_uvs = (numpy.array(scaled_uvs, dtype=numpy.float32).reshape(-1, 2)
                       if scaled_uvs else numpy.zeros((len(exp_verts), 2), dtype=numpy.float32))
    else:
        exp_verts   = raw_verts
        exp_normals = numpy.zeros_like(raw_verts)
        exp_uvs     = numpy.zeros((len(raw_verts), 2), dtype=numpy.float32)
        seq_indices = numpy.array([], dtype=numpy.uint32)

    # ── Solid VAO ────────────────────────────────────────────────────────────
    vao_solid                          = glGenVertexArrays(1)
    vbo_pos, vbo_nrm, vbo_uv, ebo_tri = glGenBuffers(4)

    glBindVertexArray(vao_solid)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, exp_verts.nbytes, exp_verts, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_nrm)
    glBufferData(GL_ARRAY_BUFFER, exp_normals.nbytes, exp_normals, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
    glBufferData(GL_ARRAY_BUFFER, exp_uvs.nbytes, exp_uvs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_tri)
    if seq_indices.nbytes > 0:
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, seq_indices.nbytes, seq_indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    # ── Wireframe VAO ─────────────────────────────────────────────────────────
    vao_wire           = glGenVertexArrays(1)
    vbo_wire, ebo_edge = glGenBuffers(2)

    glBindVertexArray(vao_wire)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_wire)
    glBufferData(GL_ARRAY_BUFFER, raw_verts.nbytes, raw_verts, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_edge)
    if raw_edges.nbytes > 0:
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, raw_edges.nbytes, raw_edges, GL_STATIC_DRAW)
    glBindVertexArray(0)

    # ── Material groups ───────────────────────────────────────────────────────
    gpu_mat_groups = (
        [(tex, tri_start * 3 * 4, tri_count * 3) for tex, tri_start, tri_count in raw_mat_groups]
        if raw_mat_groups else None
    )

    # Upload pending texture if needed
    tex_name = mdl.texture_name if isinstance(mdl.texture_name, str) else None
    if tex_name:
        pending = result.PendingTextures.get(tex_name)
        if pending:
            data, w, h, _ = pending
            _upload_texture_to_gpu(tex_name, data, w, h, mdl.tex_filter)
            del result.PendingTextures[tex_name]

    result.GPUBuffers[name] = (vao_solid, len(seq_indices),
                               vao_wire,  len(raw_edges),
                               gpu_mat_groups)

def sync_object_to_gpu(obj_name: str) -> None:
    from OpenGL.GL import glBindBuffer, glBufferData, GL_ARRAY_BUFFER, GL_STATIC_DRAW
    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == obj_name and obj_name in result.GPUBuffers:
            vao, vbo, ebo_edges, edge_count, ebo_tris, tri_count = result.GPUBuffers[obj_name]  # ← was 4-tuple
            raw_verts = numpy.array(obj[5], dtype=numpy.float32)
            if raw_verts.shape[1] == 4:
                raw_verts = raw_verts[:, :3]
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, raw_verts.nbytes, raw_verts, GL_STATIC_DRAW)
        return
#! ----------

#! Transformation functions
def apply_noise(obj_name, noise, min_height, max_height,
                height_scale=1.0, scale=0.1, octaves=4,
                persistence=0.5, lacunarity=2.0, seed=None):
    if seed is None:
        seed = random.randint(0, 10000)

    for obj in result.MainScene.ObjectsOnScene:
        if obj.name != obj_name:
            continue

        verts = numpy.array(obj.model.vertices, dtype=numpy.float32)
        xs = verts[:, 0]
        zs = verts[:, 2]

        match noise:
            case 0:
                verts[:, 1] = numpy.random.randint(min_height, max_height, size=len(verts)).astype(numpy.float32)
            case 1:
                gen = OpenSimplex(seed=seed)
                rng = numpy.random.default_rng(seed)
                seed_x = rng.uniform(0, 100)
                seed_z = rng.uniform(0, 100)
                value    = numpy.zeros(len(verts), dtype=numpy.float64)
                amp      = 1.0
                freq     = 1.0
                max_possible = 0.0
                noise_fn = numpy.vectorize(gen.noise2)
                for _ in range(octaves):
                    value        += noise_fn(xs * scale * freq + seed_x, zs * scale * freq + seed_z) * amp
                    max_possible += amp
                    amp          *= persistence
                    freq         *= lacunarity
                normalized  = (value / max_possible + 1.0) / 2.0
                verts[:, 1] = (normalized * height_scale).astype(numpy.float32)

        # Write back directly to the model's vertex list
        for i, v in enumerate(obj.model.vertices):
            v[0] = verts[i, 0]
            v[1] = verts[i, 1]
            v[2] = verts[i, 2]
        return

    print(f"apply_noise({obj_name}, ...): Object not found!")

def set_object_rotation(name: str, rx: float, ry: float, rz: float) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj.name == name:
            obj.rotation = (rx, ry, rz)
            return
    print(f"set_object_rotation: object '{name}' not found")

def rotate_object_by(name: str, drx: float, dry: float, drz: float) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj.name == name:
            rx, ry, rz = obj.rotation
            obj.rotation = (rx + drx, ry + dry, rz + drz)
            return
    print(f"rotate_object_by: object '{name}' not found")

def move_object(by_x: float, by_y: float, by_z: float, object_name: str) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj.name == object_name:
            obj.position[0] += by_x
            obj.position[1] += by_y
            obj.position[2] += by_z
            return
    print(f"move_object(..., {object_name}): The entered object does not exist!")

def set_object_texture(obj_name: str, texture_name: str,
                       tiling_x: float = 1.0, tiling_y: float = 1.0,
                       tex_filter: str = "linear") -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj.name == obj_name:
            obj.model.texture_name = texture_name
            obj.model.tiling       = (tiling_x, tiling_y)
            obj.model.tex_filter   = tex_filter
            return
    print(f"set_object_texture: object '{obj_name}' not found")
#! ----------

#! Create objects
def create_plane(name, position, sizeX, sizeZ, subdivision=1, color=(0,0,0,1), flat_shading=True):
    offset_x = sizeX / 2
    offset_z = sizeZ / 2
    steps = subdivision + 1

    vertices, edges, faces, uvs = [], [], [], []
    for row in range(steps):
        for col in range(steps):
            x =  offset_x - (col / subdivision) * sizeX
            z = -offset_z + (row / subdivision) * sizeZ
            vertices.append([x, 0, z, 1])
            uvs.append([col / subdivision, row / subdivision])

    for row in range(steps):
        for col in range(steps):
            i = row * steps + col
            if col < subdivision:
                edges.append((i, i + 1))
            if row < subdivision:
                edges.append((i, i + steps))
            if col < subdivision and row < subdivision:
                faces.append((i, i + 1, i + steps + 1, i + steps))

    triangles = _triangulate_quads(faces)

    obj = SceneObject(name, position, _normalize_color(color))
    obj.model.vertices    = vertices
    obj.model.edges       = edges
    obj.model.triangles   = triangles
    obj.model.uvs         = uvs
    obj.model.flat_shading = flat_shading
    result.MainScene.ObjectsOnScene.append(obj)

def _box_uv(x, y, z):
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        return [(z / ax + 1) / 2, (y / ax + 1) / 2]
    elif ay >= ax and ay >= az:
        return [(x / ay + 1) / 2, (z / ay + 1) / 2]
    else:
        return [(x / az + 1) / 2, (y / az + 1) / 2]

def create_cube(position, name, sizeX, sizeY, sizeZ, color=(0,0,0,1), flat_shading=True):
    hx, hy, hz = sizeX / 2, sizeY / 2, sizeZ / 2
    face_quads = [
        [( hx,-hy, hz,1),( hx, hy, hz,1),(-hx, hy, hz,1),(-hx,-hy, hz,1)],
        [(-hx,-hy,-hz,1),(-hx, hy,-hz,1),( hx, hy,-hz,1),( hx,-hy,-hz,1)],
        [( hx,-hy,-hz,1),( hx, hy,-hz,1),( hx, hy, hz,1),( hx,-hy, hz,1)],
        [(-hx,-hy, hz,1),(-hx, hy, hz,1),(-hx, hy,-hz,1),(-hx,-hy,-hz,1)],
        [( hx, hy, hz,1),( hx, hy,-hz,1),(-hx, hy,-hz,1),(-hx, hy, hz,1)],
        [( hx,-hy,-hz,1),( hx,-hy, hz,1),(-hx,-hy, hz,1),(-hx,-hy,-hz,1)],
    ]
    face_uvs = [[0,0],[0,1],[1,1],[1,0]]

    vertices, uvs, triangles, edges = [], [], [], []
    for quad in face_quads:
        base = len(vertices)
        vertices.extend([list(v) for v in quad])
        uvs.extend([uv[:] for uv in face_uvs])
        triangles.extend([base, base+1, base+2, base, base+2, base+3])
        edges.extend([(base,base+1),(base+1,base+2),(base+2,base+3),(base+3,base)])

    obj = SceneObject(name, position, _normalize_color(color))
    obj.model.vertices    = vertices
    obj.model.edges       = edges
    obj.model.triangles   = triangles
    obj.model.uvs         = uvs
    obj.model.flat_shading = flat_shading
    result.MainScene.ObjectsOnScene.append(obj)

def create_sphere(position, name, radius, segments, rings, color=(0,0,0,1), flat_shading=True):
    vertices, uvs = [], []
    theta_values = numpy.linspace(0, numpy.pi, rings)
    phi_values   = numpy.linspace(0, 2 * numpy.pi, segments)

    for theta in theta_values:
        for phi in phi_values:
            x = radius * numpy.cos(phi) * numpy.sin(theta)
            z = radius * numpy.sin(phi) * numpy.sin(theta)
            y = radius * numpy.cos(theta)
            vertices.append([x, y, z, 1.0])
            uvs.append([phi / (2 * numpy.pi), theta / numpy.pi])

    edges, triangles = [], []
    for i in range(rings):
        for j in range(segments):
            cur = i * segments + j
            edges.append((cur, i * segments + (j + 1) % segments))
            if i < rings - 1:
                edges.append((cur, (i + 1) * segments + j))

    for i in range(rings - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + j
            d = (i + 1) * segments + (j + 1) % segments
            triangles.extend([a, b, d, a, d, c])

    obj = SceneObject(name, position, _normalize_color(color))
    obj.model.vertices    = vertices
    obj.model.edges       = edges
    obj.model.triangles   = triangles
    obj.model.uvs         = uvs
    obj.model.flat_shading = flat_shading
    result.MainScene.ObjectsOnScene.append(obj)

def create_cone(position, name, radius, height, segments, base_center, color=(0,0,0,1), flat_shading=True):
    vertices = []
    uvs      = []

    # Base ring vertices — cylindrical UVs along the bottom edge
    for i in range(segments):
        angle = 2 * numpy.pi * i / segments
        vertices.append([radius * numpy.cos(angle), -height / 2, radius * numpy.sin(angle), 1.0])
        uvs.append([i / segments, 0.0])

    # Base center — disk center UV
    base_center_idx = len(vertices)
    vertices.append([0, -height / 2, 0, 1.0])
    uvs.append([0.5, 0.5])

    # Apex — tip UV
    apex_idx = len(vertices)
    vertices.append([0, height / 2, 0, 1.0])
    uvs.append([0.5, 1.0])

    edges, triangles = [], []
    for i in range(segments):
        ni = (i + 1) % segments
        edges.append((i, ni))
        edges.append((i, apex_idx))
        if base_center:
            edges.append((i, base_center_idx))
        triangles.extend([apex_idx, i, ni])
        triangles.extend([base_center_idx, ni, i])

    obj = SceneObject(name, position, _normalize_color(color))
    obj.model.vertices     = vertices
    obj.model.edges        = edges
    obj.model.triangles    = triangles
    obj.model.uvs          = uvs
    obj.model.flat_shading = flat_shading
    result.MainScene.ObjectsOnScene.append(obj)

def load_model(directory, position, name, color=(0.2, 0.2, 1.0, 1.0), scale_x=1.0, scale_y=1.0, scale_z=1.0, tex_filter="linear", flat_shading=True):
    ext = os.path.splitext(directory)[1].lower()

    if ext == ".obj":
        _load_obj(directory, position, name, color, scale_x, scale_y, scale_z, flat_shading)
    elif ext in (".gltf", ".glb"):
        _load_gltf(directory, position, name, color, scale_x, scale_y, scale_z, tex_filter, flat_shading)
    else:
        print(f"load_model: unsupported format '{ext}'")

def _append_model(position, name, color, vertices, uvs, triangles, edges,
                  scale_x=1.0, scale_y=1.0, scale_z=1.0, flat_shading=True):
    verts = numpy.array(vertices, dtype=numpy.float32)
    verts[:, 0] *= scale_x
    verts[:, 1] *= scale_y
    verts[:, 2] *= scale_z

    obj = SceneObject(name, position, list(color))
    obj.model.vertices    = verts.tolist()
    obj.model.edges       = edges
    obj.model.triangles   = triangles
    obj.model.uvs         = uvs
    obj.model.flat_shading = flat_shading
    result.MainScene.ObjectsOnScene.append(obj)

def _load_obj(directory, position, name, color, scale_x=1.0, scale_y=1.0, scale_z=1.0, flat_shading=True):
    import re
    raw_positions = []
    raw_uvs_list  = []
    vertices      = []
    uvs           = []
    edges         = []
    triangles     = []
    vert_cache    = {}

    try:
        with open(directory, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line.startswith("v "):
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', line)
                    if len(nums) == 3:
                        raw_positions.append([float(n) * 30 for n in nums] + [1.0])
                elif line.startswith("vt "):
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', line)
                    if len(nums) >= 2:
                        raw_uvs_list.append([float(nums[0]), float(nums[1])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    face_indices = []
                    for p in parts:
                        tokens  = p.split('/')
                        pos_idx = int(tokens[0]) - 1
                        uv_idx  = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1] else -1
                        key     = (pos_idx, uv_idx)
                        if key not in vert_cache:
                            vert_cache[key] = len(vertices)
                            vertices.append(raw_positions[pos_idx])
                            uvs.append(raw_uvs_list[uv_idx] if 0 <= uv_idx < len(raw_uvs_list) else [0.0, 0.0])
                        face_indices.append(vert_cache[key])
                    for i in range(len(face_indices)):
                        edges.append((face_indices[i], face_indices[(i + 1) % len(face_indices)]))
                    for i in range(1, len(face_indices) - 1):
                        triangles.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
    except Exception as e:
        print(f"load_model({directory}, ...): {e}")
        return

    _append_model(position, name, color, vertices, uvs, triangles, edges, scale_x, scale_y, scale_z, flat_shading)

def _load_gltf(directory, position, name, color, scale_x=1.0, scale_y=1.0, scale_z=1.0, tex_filter="linear", flat_shading=True):
    try:
        from pygltflib import GLTF2
        from PIL import Image
        import struct, base64, io

        gltf = GLTF2().load(directory)

        def _blob():
            if not gltf.buffers:
                return b""
            buf = gltf.buffers[0]
            if buf.uri is None:
                return gltf.binary_blob()
            elif buf.uri.startswith("data:"):
                _, encoded = buf.uri.split(",", 1)
                return base64.b64decode(encoded)
            bin_path = os.path.join(os.path.dirname(directory), buf.uri)
            with open(bin_path, "rb") as f:
                return f.read()

        blob = _blob()

        _COMP  = {5120:"b", 5121:"B", 5122:"h", 5123:"H", 5125:"I", 5126:"f"}
        _COUNT = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

        def _read_accessor(idx):
            acc  = gltf.accessors[idx]
            bv   = gltf.bufferViews[acc.bufferView]
            fmt  = _COMP[acc.componentType]
            n    = _COUNT[acc.type]
            size = struct.calcsize(fmt) * n
            bv_off  = bv.byteOffset  or 0
            acc_off = acc.byteOffset or 0
            stride  = bv.byteStride  or size
            start   = bv_off + acc_off
            data = []
            for i in range(acc.count):
                chunk = blob[start + i * stride : start + i * stride + size]
                row   = list(struct.unpack(f"{n}{fmt}", chunk))
                data.append(row if n > 1 else row[0])
            return data

        def _load_image(img_idx) -> str:
            tex_name = f"{name}_tex_{img_idx}"
            if tex_name in result.Textures or tex_name in result.PendingTextures:
                return tex_name

            img_info = gltf.images[img_idx]
            pil_img  = None

            if img_info.bufferView is not None:
                # Embedded binary (GLB) — unchanged
                bv        = gltf.bufferViews[img_info.bufferView]
                offset    = bv.byteOffset or 0
                img_bytes = blob[offset : offset + bv.byteLength]
                pil_img   = Image.open(io.BytesIO(img_bytes))
            elif img_info.uri:
                if img_info.uri.startswith("data:"):
                    # Base64 embedded — unchanged
                    _, encoded = img_info.uri.split(",", 1)
                    pil_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
                else:
                    # External file — look in assets/textures/ instead of model directory
                    filename = os.path.basename(img_info.uri)
                    img_path = os.path.join(get_assets_path(), "assets", "textures", filename)
                    pil_img  = Image.open(img_path)

            if pil_img:
                pil_img = pil_img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
                data    = numpy.array(pil_img, dtype=numpy.uint8)
                result.PendingTextures[tex_name] = (data, pil_img.width, pil_img.height, tex_filter)

            return tex_name

        all_verts  = []
        all_uvs    = []
        all_tris   = []
        all_edges  = []
        mat_groups = []  # (tex_name_or_None, tri_start, tri_count)

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                base      = len(all_verts)
                tri_start = len(all_tris) // 3

                positions = _read_accessor(prim.attributes.POSITION)
                for p in positions:
                    all_verts.append([p[0] * 30, p[1] * 30, p[2] * 30, 1.0])

                if prim.attributes.TEXCOORD_0 is not None:
                    raw_uvs = _read_accessor(prim.attributes.TEXCOORD_0)
                    all_uvs.extend([[u, 1.0 - v] for u, v in raw_uvs])
                else:
                    all_uvs.extend([[0.0, 0.0]] * len(positions))

                indices = (_read_accessor(prim.indices)
                           if prim.indices is not None else list(range(len(positions))))

                for i in range(0, len(indices) - 2, 3):
                    a, b, c = indices[i]+base, indices[i+1]+base, indices[i+2]+base
                    all_tris.extend([a, b, c])
                    all_edges.extend([(a,b),(b,c),(c,a)])

                tri_count = len(all_tris) // 3 - tri_start

                # Resolve material → texture
                tex_name = None
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    pbr = mat.pbrMetallicRoughness
                    if pbr and pbr.baseColorTexture is not None:
                        img_idx  = gltf.textures[pbr.baseColorTexture.index].source
                        tex_name = _load_image(img_idx)

                mat_groups.append((tex_name, tri_start, tri_count))

        pos = numpy.array([position[0], position[1], position[2], 1.0])
        _append_model(position, name, color,
                  numpy.array(all_verts, dtype=numpy.float32),
                  all_uvs, all_tris, all_edges,
                  scale_x, scale_y, scale_z, flat_shading)
        result.MainScene.ObjectsOnScene[-1].model.texture_name = mat_groups

    except Exception as e:
        import traceback
        print(f"load_model GLTF ({directory}): {e}")
        traceback.print_exc()
#! ----------

#! Edit mode

#! ----------