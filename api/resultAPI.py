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

#! Result classes
class Object:
    def __init__(self) -> None:
        self.Position = [0, 0, 0]

class Camera(Object):
    def __init__(self) -> None:
        super().__init__()

        self.Yaw = 0
        self.Pitch = 0
        self.Roll = 0
        self.Fov = 110

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

        self.VertexBuffer = []
        self.EdgeBuffer = []
        self.GPUBuffers = {}
        self.RenderList = []
        self.Textures = {}
        self.PendingTextures = {}
        self.ObjectRotations = {}
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
#! ----------

#! Rendering functions
def render_scene() -> None:
    result.RenderList.clear()
    VIEW = result.Matrices.getView_matrix()
    PROJ = result.Matrices.ProjMatrix

    for obj in result.MainScene.ObjectsOnScene:
        name     = obj[0]
        position = obj[1]
        color    = obj[8] if len(obj) > 8 else [1.0, 1.0, 1.0, 1.0]

        TRANS = result.Matrices.getTrans_matrix(*position[:3])

        # Apply stored per-object rotation (no vertex mutation needed)
        rot = result.ObjectRotations.get(name, (0.0, 0.0, 0.0))
        ROT = (result.Matrices.getxRot_matrix(rot[0])
             @ result.Matrices.getyRot_matrix(rot[1])
             @ result.Matrices.getzRot_matrix(rot[2]))

        MVP = numpy.ascontiguousarray((ROT @ TRANS @ VIEW @ PROJ), dtype=numpy.float32)
        result.RenderList.append((name, MVP, color))

def load_texture(texture_name: str, path: str) -> None:
    """
    Call this from main() before the render loop.
    The actual GL upload happens lazily inside paintGL where the context is current.
    """
    from PIL import Image
    try:
        img  = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        data = numpy.array(img, dtype=numpy.uint8)
        result.PendingTextures[texture_name] = (data, img.width, img.height)
    except Exception as e:
        print(f"load_texture({texture_name}): {e}")

def _upload_texture_to_gpu(texture_name: str, data, width: int, height: int) -> None:
    from OpenGL.GL import (glGenTextures, glBindTexture, glTexImage2D,
                           glTexParameteri, glGenerateMipmap,
                           GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE,
                           GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
                           GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                           GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_REPEAT)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)

    result.Textures[texture_name] = tex_id

def upload_object_to_gpu(obj) -> None:
    from OpenGL.GL import (glGenVertexArrays, glGenBuffers, glBindVertexArray,
                           glBindBuffer, glBufferData, glEnableVertexAttribArray,
                           glVertexAttribPointer, GL_ARRAY_BUFFER,
                           GL_ELEMENT_ARRAY_BUFFER, GL_FLOAT, GL_FALSE,
                           GL_STATIC_DRAW)

    name      = obj[0]
    raw_verts = numpy.array(obj[5], dtype=numpy.float32)
    if raw_verts.shape[1] == 4:
        raw_verts = raw_verts[:, :3]

    raw_edges = numpy.array(obj[6], dtype=numpy.uint32).ravel()
    raw_tris  = (numpy.array(obj[7], dtype=numpy.uint32).ravel()
                 if len(obj) > 7 and obj[7] else numpy.array([], dtype=numpy.uint32))
    raw_uvs   = obj[9] if len(obj) > 9 and obj[9] is not None else None

    # --- Solid VAO (expanded for flat shading) ---
    if len(raw_tris) > 0:
        exp_verts, exp_normals, exp_uvs, seq_indices = \
            _expand_for_flat_shading(obj[5], obj[7], raw_uvs)
    else:
        exp_verts   = raw_verts
        exp_normals = numpy.zeros_like(raw_verts)
        exp_uvs     = numpy.zeros((len(raw_verts), 2), dtype=numpy.float32)
        seq_indices = numpy.array([], dtype=numpy.uint32)

    vao_solid                            = glGenVertexArrays(1)
    vbo_pos, vbo_nrm, vbo_uv, ebo_tris  = glGenBuffers(4)

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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_tris)
    if seq_indices.nbytes > 0:
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, seq_indices.nbytes, seq_indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    # --- Wireframe VAO (original shared verts, no normals/UVs needed) ---
    vao_wire           = glGenVertexArrays(1)
    vbo_wire, ebo_edge = glGenBuffers(2)

    glBindVertexArray(vao_wire)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_wire)
    glBufferData(GL_ARRAY_BUFFER, raw_verts.nbytes, raw_verts, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_edge)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, raw_edges.nbytes, raw_edges, GL_STATIC_DRAW)

    glBindVertexArray(0)

    result.GPUBuffers[name] = (vao_solid, len(seq_indices),
                               vao_wire,  len(raw_edges))

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
def apply_noise(obj_name: str, noise: int, min_height: int, max_height: int,
                height_scale: float = 1.0,
                scale: float = 0.1,
                octaves: int = 4,
                persistence: float = 0.5,
                lacunarity: float = 2.0,
                seed: int = None) -> None:

    if seed is None:
        seed = random.randint(0, 10000)

    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] != obj_name:
            continue

        verts = numpy.array(obj[5], dtype=numpy.float32)  # (N, 4)
        xs = verts[:, 0]
        zs = verts[:, 2]

        match noise:
            case 0:
                # Vectorized random noise
                verts[:, 1] = numpy.random.randint(min, max, size=len(verts)).astype(numpy.float32)

            case 1:
                gen = OpenSimplex(seed=seed)
                rng = numpy.random.default_rng(seed)
                seed_x = rng.uniform(0, 100)
                seed_z = rng.uniform(0, 100)

                # Vectorized fBm — no Python loop over vertices
                value    = numpy.zeros(len(verts), dtype=numpy.float64)
                amp      = 1.0
                freq     = 1.0
                max_possible = 0.0

                noise_fn = numpy.vectorize(gen.noise2)

                for _ in range(octaves):
                    x_coords = xs * scale * freq + seed_x
                    z_coords = zs * scale * freq + seed_z

                    value        += noise_fn(x_coords, z_coords) * amp
                    max_possible += amp
                    amp          *= persistence
                    freq         *= lacunarity

                # Normalize to [0, 1] then scale
                normalized   = (value / max_possible + 1.0) / 2.0
                verts[:, 1]  = (normalized * height_scale).astype(numpy.float32)

        # Write back
        for i, v in enumerate(obj[5]):
            v[0] = verts[i, 0]
            v[1] = verts[i, 1]
            v[2] = verts[i, 2]
            v[3] = verts[i, 3]

        return

    print(f"apply_noise({obj_name}, ...): Object not found!")

def set_object_rotation(name: str, rx: float, ry: float, rz: float) -> None:
    """Set absolute rotation (degrees) for an object. No vertex mutation."""
    result.ObjectRotations[name] = (rx, ry, rz)

def rotate_object_by(name: str, drx: float, dry: float, drz: float) -> None:
    """Increment rotation (degrees) for an object each tick. Use this in update_scene()."""
    rx, ry, rz = result.ObjectRotations.get(name, (0.0, 0.0, 0.0))
    result.ObjectRotations[name] = (rx + drx, ry + dry, rz + drz)

def move_object(by_x: float, by_y: float, by_z: float, object_name: str) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == object_name:
            obj[1] @= result.Matrices.getTrans_matrix(by_x, by_y, by_z)
            return
    
    print(f"move_object({by_x}, {by_y}, {by_z}, {object_name}): The entered object does not exist!")

def set_object_texture(obj_name: str, texture_name: str) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == obj_name:
            while len(obj) < 11:
                obj.append(None)
            obj[10] = texture_name
            return
    print(f"set_object_texture: object '{obj_name}' not found")
#! ----------

#! Create objects
def create_plane(name: str, position: tuple, sizeX: float, sizeZ: float, subdivision: int, color=(0.0, 0.0, 0.0, 1.0)) -> None:
    offset_x = sizeX / 2
    offset_z = sizeZ / 2

    steps = subdivision + 1  # number of vertices per side

    vertices = []
    for row in range(steps):
        for col in range(steps):
            x =  offset_x - (col / subdivision) * sizeX
            z = -offset_z + (row / subdivision) * sizeZ
            vertices.append([x, 0, z, 1])

    # Build edges and faces from the grid
    edges = []
    faces = []

    for row in range(steps):
        for col in range(steps):
            i = row * steps + col

            # Horizontal edge (connect to the right neighbour)
            if col < subdivision:
                edges.append((i, i + 1))

            # Vertical edge (connect to the neighbour below)
            if row < subdivision:
                edges.append((i, i + steps))

            # Face (quad) from top-left corner of each cell
            if col < subdivision and row < subdivision:
                top_left     = i
                top_right    = i + 1
                bottom_left  = i + steps
                bottom_right = i + steps + 1
                faces.append((top_left, top_right, bottom_right, bottom_left))
    triangles = _triangulate_quads(faces)

    uvs = []
    for row in range(steps):
        for col in range(steps):
            uvs.append([col / subdivision, row / subdivision])

    pos = numpy.array([position[0], position[1], position[2], 1])
    result.MainScene.ObjectsOnScene.append(
        [name, pos, sizeX, faces, sizeZ, vertices, edges, triangles, list(color), uvs, None]
    )

def _box_uv(x, y, z):
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        return [(z / ax + 1) / 2, (y / ax + 1) / 2]
    elif ay >= ax and ay >= az:
        return [(x / ay + 1) / 2, (z / ay + 1) / 2]
    else:
        return [(x / az + 1) / 2, (y / az + 1) / 2]

def create_cube(position: tuple, name: str, sizeX: float, sizeY: float,
                sizeZ: float, color=(0.0, 0.0, 0.0, 1.0)) -> None:
    offset_x = sizeX / 2
    offset_y = sizeY / 2
    offset_z = sizeZ / 2

    vertices = [
        ( offset_x,  offset_y,  offset_z, 1),
        ( offset_x, -offset_y,  offset_z, 1),
        (-offset_x, -offset_y,  offset_z, 1),
        (-offset_x,  offset_y,  offset_z, 1),
        ( offset_x,  offset_y, -offset_z, 1),
        ( offset_x, -offset_y, -offset_z, 1),
        (-offset_x, -offset_y, -offset_z, 1),
        (-offset_x,  offset_y, -offset_z, 1),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    faces = [
        (0,1,2,3), (4,7,6,5),  # front, back
        (0,4,5,1), (3,2,6,7),  # right, left
        (0,3,7,4), (1,5,6,2),  # top, bottom
    ]
    triangles = _triangulate_quads(faces)
    
    uvs = [_box_uv(x, y, z) for (x, y, z, _) in vertices]

    position = numpy.array([position[0], position[1], position[2], 1])
    result.MainScene.ObjectsOnScene.append(
        [name, position, sizeX, sizeY, sizeZ, vertices, edges, triangles, list(color), uvs, None]
    )

def create_sphere(position: tuple, name: str, radius: float, segments: int, rings: int, color=(0.0, 0.0, 0.0, 1.0)) -> None:
    num_vertices = rings * segments
    vertices = numpy.zeros((num_vertices, 4), dtype=numpy.float32)
    
    theta_values = numpy.linspace(0, numpy.pi, rings)
    phi_values = numpy.linspace(0, 2 * numpy.pi, segments)
    
    idx = 0
    for i, theta in enumerate(theta_values):
        sin_theta = numpy.sin(theta)
        cos_theta = numpy.cos(theta)
        
        for j, phi in enumerate(phi_values):
            x = radius * numpy.cos(phi) * sin_theta
            z = radius * numpy.sin(phi) * sin_theta
            y = radius * cos_theta
            
            vertices[idx] = [x, y, z, 1.0]
            idx += 1
    
    num_edges = rings * segments + (rings - 1) * segments
    edges = numpy.zeros((num_edges, 2), dtype=numpy.int32)
    
    edge_idx = 0
    
    for i in range(rings):
        for j in range(segments):
            current = i * segments + j
            
            right = i * segments + (j + 1) % segments
            edges[edge_idx] = [current, right]
            edge_idx += 1
            
            if i < rings - 1:
                below = (i + 1) * segments + j
                edges[edge_idx] = [current, below]
                edge_idx += 1
    
    tris = []
    for i in range(rings - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + j
            d = (i + 1) * segments + (j + 1) % segments
            tris.extend([a, b, d, a, d, c])

    uvs = []
    for i, theta in enumerate(theta_values):
        for j, phi in enumerate(phi_values):
            uvs.append([phi / (2 * numpy.pi), theta / numpy.pi])

    position = numpy.array([position[0], position[1], position[2], 1.0], dtype=numpy.float32)
    result.MainScene.ObjectsOnScene.append(
        [name, position, radius, segments, rings, vertices, edges, tris, list(color), uvs, None]
    )

def create_cone(position: tuple, name: str, radius: float, height: float,
                segments: int, base_center: bool, color=(0.0, 0.0, 0.0, 1.0)) -> None:
    vertices = []

    # Base ring vertices
    for segment in range(segments):
        angle = 2 * numpy.pi * segment / segments
        x = radius * numpy.cos(angle)
        z = radius * numpy.sin(angle)
        vertices.append((x, -height / 2, z, 1.0))

    # Append special vertices first so indices are stable
    base_center_idx = len(vertices)
    vertices.append((0, -height / 2, 0, 1.0))  # base center, same Y as ring

    apex_idx = len(vertices)
    vertices.append((0, height / 2, 0, 1.0))   # apex, symmetric above base

    # Edges
    edges = []
    for i in range(segments):
        next_i = (i + 1) % segments
        edges.append((i, next_i))               # base ring
        edges.append((i, apex_idx))             # side edge to apex
        if base_center:
            edges.append((i, base_center_idx))  # spoke to base center

    # Triangles
    tris = []
    for i in range(segments):
        next_i = (i + 1) % segments
        tris.extend([apex_idx, i, next_i])          # side face
        tris.extend([base_center_idx, next_i, i])   # base face

    position = numpy.array([position[0], position[1], position[2], 1.0])
    result.MainScene.ObjectsOnScene.append(
        [name, position, radius, height, segments, vertices, edges, tris, list(_normalize_color(color))]
    )

def load_model(directory, position, name, color=(0.2, 0.2, 1.0, 1.0)):
    import re
    raw_positions = []
    raw_uvs_list  = []
    vertices      = []
    uvs           = []
    edges         = []
    triangles     = []
    vert_cache    = {}   # (pos_idx, uv_idx) -> combined index

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
                        tokens   = p.split('/')
                        pos_idx  = int(tokens[0]) - 1
                        uv_idx   = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1] else -1
                        key      = (pos_idx, uv_idx)
                        if key not in vert_cache:
                            vert_cache[key] = len(vertices)
                            vertices.append(raw_positions[pos_idx])
                            if uv_idx >= 0 and uv_idx < len(raw_uvs_list):
                                uvs.append(raw_uvs_list[uv_idx])
                            else:
                                uvs.append([0.0, 0.0])
                        face_indices.append(vert_cache[key])

                    for i in range(len(face_indices)):
                        edges.append((face_indices[i], face_indices[(i+1) % len(face_indices)]))
                    for i in range(1, len(face_indices) - 1):
                        triangles.extend([face_indices[0], face_indices[i], face_indices[i+1]])

    except Exception as e:
        print(f"load_model({directory}): {e}")
        return

    vertices = numpy.array(vertices, dtype=numpy.float32)
    position = numpy.array([position[0], position[1], position[2], 1.0])
    result.MainScene.ObjectsOnScene.append(
        [name, position, None, None, None, vertices, edges, triangles, list(color), uvs, None]
    )
#! ----------

#! Edit mode

#! ----------

#! Character control functions

#! ----------