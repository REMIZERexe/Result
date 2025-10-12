import sys

sys.dont_write_bytecode = True
from configparser import ConfigParser
config = ConfigParser()
import os
config.read(os.path.join(os.path.dirname(__file__), "app/config.ini"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Callable
import math
import numpy

def timer(func: Callable) -> None:
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
        self.far = 200.0
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
        # Note: We're building the INVERSE of the camera's transform
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
    for obj in result.MainScene.ObjectsOnScene:
        name, position, sizeX, sizeY, sizeZ, vertices, edges = obj
        clipped_edges = project(vertices, position, edges)

        for edge in clipped_edges:
            start, end = edge
            
            engine_window.draw_line(start, end)

def project(vertices: list, position: tuple, edges: list) -> list:
    TRANS = result.Matrices.getTrans_matrix(*position[:3])
    VIEW = result.Matrices.getView_matrix()
    
    vertices_view = []
    for vertex in vertices:
        if len(vertex) == 3:
            vertex = numpy.array([*vertex, 1.0])
        vertex_view = vertex @ TRANS @ VIEW
        vertices_view.append(vertex_view)
    
    clipped_lines = []
    near_plane = config.getfloat("camera", "near_plane")
    
    for edge in edges:
        start_idx, end_idx = edge
        v1 = vertices_view[start_idx]
        v2 = vertices_view[end_idx]
        
        z1, z2 = v1[2], v2[2]
        
        if z1 > near_plane and z2 > near_plane:
            continue
        
        if z1 > near_plane or z2 > near_plane:
            # Calculate interpolation factor where line crosses near plane
            t = (near_plane - z1) / (z2 - z1)
            
            clipped_vertex = v1 + t * (v2 - v1)
            
            if z1 > near_plane:
                v1 = clipped_vertex
            else:
                v2 = clipped_vertex
        
        proj1 = v1 @ result.Matrices.ProjMatrix
        proj2 = v2 @ result.Matrices.ProjMatrix
        
        if proj1[3] != 0:
            proj1 /= proj1[3]
        if proj2[3] != 0:
            proj2 /= proj2[3]
        screen1 = ndc_to_screen(proj1)
        screen2 = ndc_to_screen(proj2)
        
        clipped_lines.append((screen1, screen2))
    
    return clipped_lines

def ndc_to_screen(point: list) -> tuple:
    x_ndc, y_ndc = point[0], point[1]
    screen_x = int((x_ndc + 1) * config.getint("window", "width") / 2)
    screen_y = int((1 - y_ndc) * config.getint("window", "height") / 2)
    return (screen_x, screen_y)
#! ----------

#! Transformation functions
def rotate_object(axis: str, object_name: str, angle: float) -> None:
    if axis.upper() != "X" and axis.upper() != "Y" and axis.upper() != "Z":
        print(f"rotate_object({axis}, {object_name}, {angle}): You can't rotate an object on a non-existent axis!")
        return
    
    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == object_name:
            vertices = obj[5]
            rotated = []

            for vertex in vertices:
                match axis.upper():
                    case "X":
                        rotated.append(result.Matrices.getxRot_matrix(angle) @ vertex)
                    case "Y":
                        rotated.append(result.Matrices.getyRot_matrix(angle) @ vertex)
                    case "Z":
                        rotated.append(result.Matrices.getzRot_matrix(angle) @ vertex)

            obj[5] = rotated
            return
    
    print(f"rotate_object({axis}, {object_name}, {angle}): The entered object does not exist!")

def move_object(by_x: float, by_y: float, by_z: float, object_name: str) -> None:
    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == object_name:
            obj[1] @= result.Matrices.getTrans_matrix(by_x, by_y, by_z)
            return
    
    print(f"move_object({by_x}, {by_y}, {by_z}, {object_name}): The entered object does not exist!")
#! ----------

#! Create objects
def create_instance(obj_name: str, inst_name: str, position: tuple) -> None:
    if len(position) > 3 or len(position) < 3:
        print(f"create_instance({obj_name}, {inst_name}, {position}): The entered position is invalid!")

    for obj in result.MainScene.ObjectsOnScene:
        if obj[0] == obj_name:
            result.MainScene.ObjectsOnScene.append((inst_name, position, obj[2], obj[3], obj[4], obj[5], obj[6]))
            return
    
    print(f"create_instance({obj_name}, {inst_name}, {position}): The entered object does not exist!")

def create_cube(position: tuple, name: str, sizeX: float, sizeY: float, sizeZ: float) -> None:
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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    position = numpy.array([position[0], position[1], position[2], 1])

    result.MainScene.ObjectsOnScene.append([name, position, sizeX, sizeY, sizeZ, vertices, edges])

def create_sphere(position: tuple, name: str, radius: float, segments: int, rings: int) -> None:
    vertices = []
    edges = []

    for i in range(rings):
        theta = math.pi * i / (rings - 1)
        for j in range(segments):
            phi = 2 * math.pi * j / segments

            x = radius * math.cos(phi) * math.sin(theta)
            z = radius * math.sin(phi) * math.sin(theta)
            y = radius * math.cos(theta)

            vertices.append((x, y, z, 1))
    
    for i in range(rings):
        for j in range(segments):
            current = i * segments + j
            right = i * segments + (j + 1) % segments
            if j < segments - 1 or True:
                edges.append((current, right))

            if i < rings - 1:
                below = (i + 1) * segments + j
                edges.append((current, below))
                numpy.array(edges)

    position = numpy.array([position[0], position[1], position[2], 1.0])

    result.MainScene.ObjectsOnScene.append([name, position, radius, segments, rings, vertices, edges])

def create_cone(position: tuple, name: str, radius: float, height: float, segments: int, base_center: bool) -> None:
    vertices = []
    edges = []

    apex = (0, height, 0, 1.0)
    base_center_vertex = (0, -(height / 2), 0, 1.0)

    for segment in range(segments):
        angle = 2 * numpy.pi * segment / segments
        x = radius * numpy.cos(angle)
        z = radius * numpy.sin(angle)
        y = 0
        vertices.append((x, y - height / 2, z, 1.0))

    for vertex in range(0, len(vertices)):
        if base_center:
            edges.append((vertex, len(vertices)))

        edges.append((vertex, len(vertices) + 1))

        if vertex != segments - 1:
            edges.append((vertex, vertex + 1))
    edges.append((0, len(vertices) - 1))

    vertices.append(base_center_vertex)
    vertices.append(apex)
    position = numpy.array([position[0], position[1], position[2], 1.0])

    result.MainScene.ObjectsOnScene.append([name, position, radius, height, segments, vertices, edges])

def load_model(directory: str, position: tuple, name: str) -> None:
    import re
    vertices = []
    edges = []

    try:
        with open(directory, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line.startswith("v "):
                    vertex_raw = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', line)
                    if len(vertex_raw) == 3:
                        vertex = [float(coord) * 30 for coord in vertex_raw]
                        vertex.append(1.0)
                        vertices.append(vertex)
                    else:
                        print(f"load_model({directory}, {position}, {name}): Skiped vertex: Incorrect vertex format! Only 3 dimensional vertices are accepted!")

                elif line.startswith("f "):
                    try:
                        parts = line.split()[1:]
                        vertex_indices = [int(p.split('/')[0]) - 1 for p in parts]

                        for i in range(len(vertex_indices)):
                            start = vertex_indices[i]
                            end = vertex_indices[(i + 1) % len(vertex_indices)]
                            edges.append((start, end))
                    except Exception as e:
                        print(f"load_model({directory}, {position}, {name}): Incorrect edge or face format!: {e}")
    except Exception as e:
        print(f"load_model({directory}, {position}, {name}): File not found at this directory!")
        return

    try:
        vertices = numpy.array(vertices, dtype=numpy.float32)
    except Exception as e:
        print(f"load_model({directory}, {position}, {name}): Failed to create a numpy array of vertices!: {e}")
        return

    position = numpy.array([position[0], position[1], position[2], 1.0])

    result.MainScene.ObjectsOnScene.append([name, position, None, None, None, vertices, edges])
#! ----------

#! Edit mode

#! ----------

#! Character control functions

#! ----------