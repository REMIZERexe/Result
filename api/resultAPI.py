import sys
sys.dont_write_bytecode = True
import os
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
class WindowSettings:
    def __init__(self) -> None:
        self.Fullscreen = False
        self.WindowSize = {
            "width": 1280,
            "height": 720
        }

class Object:
    def __init__(self) -> None:
        self.Position = numpy.array([0, 0, 0])

class Camera(Object):
    def __init__(self) -> None:
        super().__init__()

        self.DefYaw = 0
        self.DefPitch = 0
        self.Fov = 90

class Scene:
    def __init__(self) -> None:
        self.ObjectsOnScene = []
        self.MainCamera = None

class Matrices:
    def __init__(self) -> None:
        class MatriceSettings:
            aspect_ratio = Result.WindowParam.WindowSize["width"] / Result.WindowParam.WindowSize["height"]
            near = 0.1
            far = 200.0
            f = 1 / math.tan(math.radians(Result.MainScene.MainCamera.Fov) / 2)

        self.ProjMatrix = numpy.array([
            [MatriceSettings.f / MatriceSettings.aspect_ratio, 0, 0, 0],
            [0, MatriceSettings.f, 0, 0],
            [0, 0, (MatriceSettings.far + MatriceSettings.near) / (MatriceSettings.near - MatriceSettings.far), (2 * MatriceSettings.far * MatriceSettings.near) / (MatriceSettings.near - MatriceSettings.far)],
            [0, 0, -1, 0]
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
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        return translation_matrix
    
class Result:
    def __init__(self) -> None:
        self.MainScene = None
        self.Matrices = None
        self.WindowParam = None

        self.VertexBuffer = []
        self.EdgeBuffer = []
#! ----------

#! Settings by default
def set_window_instance(win) -> None:
    global engine_window
    engine_window = win

def set_scene_instance(scene) -> None:
    Result.MainScene = scene

def set_camera_instance(camera) -> None:
    Result.MainScene.MainCamera = camera

def set_matrices_instance(instance) -> None:
    Result.Matrices = instance

def set_window_settings_instance(instance) -> None:
    Result.WindowParam = instance
#! ----------

#! Rendering functions
def render_scene() -> None:
    for obj in Result.MainScene.ObjectsOnScene:
        name, position, sizeX, sizeY, sizeZ, vertices, edges = obj
        points = project_points(vertices, position)

        for edge in edges:
            engine_window.draw_line(points[edge[0]], points[edge[1]])

def project_points(vertices, position) -> numpy.ndarray:
    proj_points = []
    for vertex in vertices:
        if len(vertex) == 3:
            vertex = numpy.array([*vertex, 1.0])
        if len(Result.MainScene.MainCamera.Position) == 3:
            Result.MainScene.MainCamera.position = numpy.array([*Result.MainScene.MainCamera.Position, 1.0])

        vertex = Result.Matrices.getTrans_matrix(*position[:3]) @ vertex
        vertex = Result.Matrices.getTrans_matrix(-Result.MainScene.MainCamera.Position[0], -Result.MainScene.MainCamera.Position[1], -Result.MainScene.MainCamera.Position[2]) @ vertex
        vertex = Result.Matrices.getyRot_matrix(Result.MainScene.MainCamera.DefYaw) @ vertex
        vertex = Result.Matrices.getxRot_matrix(Result.MainScene.MainCamera.DefPitch) @ vertex

        proj = Result.Matrices.ProjMatrix @ vertex

        if proj[3] != 0:
            proj /= proj[3]

        proj = ndc_to_screen(proj)

        proj_points.append(proj)

    return numpy.array(proj_points)

def ndc_to_screen(point) -> tuple:
    x_ndc, y_ndc = point[0], point[1]
    screen_x = int((x_ndc + 1) * Result.WindowParam.WindowSize["width"] / 2)
    screen_y = int((1 - y_ndc) * Result.WindowParam.WindowSize["height"] / 2)
    return (screen_x, screen_y)

def clip_to_screen(p1, p2, width, height):
    def is_inside(p):
        return 0 <= p[0] < width and 0 <= p[1] < height

    if is_inside(p1) and is_inside(p2):
        return p1, p2  # Уже внутри

    def intersect(p_in, p_out):
        x1, y1 = p_in
        x2, y2 = p_out
        dx = x2 - x1
        dy = y2 - y1

        if dx != 0:
            if x2 < 0:
                t = (0 - x1) / dx
            elif x2 > width:
                t = (width - 1 - x1) / dx
            else:
                t = 1
        elif dy != 0:
            if y2 < 0:
                t = (0 - y1) / dy
            elif y2 > height:
                t = (height - 1 - y1) / dy
            else:
                t = 1
        else:
            t = 1

        new_x = x1 + t * dx
        new_y = y1 + t * dy
        return (new_x, new_y)

    if is_inside(p1):
        p2 = intersect(p1, p2)
    elif is_inside(p2):
        p1 = intersect(p2, p1)
    else:
        return None  # обе вне — не рисуем

    return p1, p2
#! ----------

#! Transformation functions
def rotate_object(axis: str, object_name: str, angle: float) -> None:
    if axis.upper() != "X" and axis.upper() != "Y" and axis.upper() != "Z":
        print(f"rotate_object({axis}, {object_name}, {angle}): You can't rotate an object on a non-existent axis!")
        return
    
    for obj in Result.MainScene.ObjectsOnScene:
        if obj[0] == object_name:
            vertices = obj[5]
            rotated = []

            for vertex in vertices:
                match axis.upper():
                    case "X":
                        rotated.append(Result.Matrices.getxRot_matrix(angle) @ vertex)
                    case "Y":
                        rotated.append(Result.Matrices.getyRot_matrix(angle) @ vertex)
                    case "Z":
                        rotated.append(Result.Matrices.getzRot_matrix(angle) @ vertex)

            obj[5] = rotated
            return
    
    print(f"rotate_object({axis}, {object_name}, {angle}): The entered object does not exist!")

def move_object(by_x: float, by_y: float, by_z: float, object_name: str) -> None:
    for obj in Result.MainScene.ObjectsOnScene:
        if obj[0] == object_name:
            obj[1] = (Result.Matrices.getTrans_matrix(by_x, by_y, by_z) @ obj[1])
            return
    
    print(f"move_object({by_x}, {by_y}, {by_z}, {object_name}): The entered object does not exist!")
#! ----------

#! Create objects
def create_instance(obj_name: str, inst_name: str, position: tuple) -> None:
    if len(position) > 3 or len(position) < 3:
        print(f"create_instance({obj_name}, {inst_name}, {position}): The entered position is invalid!")

    for obj in Result.MainScene.ObjectsOnScene:
        if obj[0] == obj_name:
            Result.MainScene.ObjectsOnScene.append((inst_name, position, obj[2], obj[3], obj[4], obj[5], obj[6]))
            return
    
    print(f"create_instance({obj_name}, {inst_name}, {position}): The entered object does not exist!")

def create_cube(position: tuple, name: str, sizeX: float, sizeY: float, sizeZ: float) -> None:
    offset_x = sizeX / 2
    offset_y = sizeY / 2
    offset_z = sizeZ / 2

    vertices = numpy.array([
        ( offset_x,  offset_y,  offset_z, 1),
        ( offset_x, -offset_y,  offset_z, 1),
        (-offset_x, -offset_y,  offset_z, 1),
        (-offset_x,  offset_y,  offset_z, 1),
        ( offset_x,  offset_y, -offset_z, 1),
        ( offset_x, -offset_y, -offset_z, 1),
        (-offset_x, -offset_y, -offset_z, 1),
        (-offset_x,  offset_y, -offset_z, 1),
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    position = [position[0], position[1], position[2], 1]

    Result.MainScene.ObjectsOnScene.append([name, position, sizeX, sizeY, sizeZ, vertices, edges])

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
            numpy.array(vertices)
    
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

    position = [position[0], position[1], position[2], 1.0]

    Result.MainScene.ObjectsOnScene.append([name, position, radius, segments, rings, vertices, edges])

def create_cone(position: tuple, name: str, radius: float, height: float, segments: int) -> None:
    vertices = []
    edges = []

    apex = (0, height, 0)
    base_center = (0, 0, 0)

    for i in range(segments):
        angle = 2 * numpy.pi * i / segments
        x = radius * numpy.cos(angle)
        z = radius * numpy.sin(angle)
        y = 0
        vertices.append((x, y, z))

    for i in range(0, len(vertices)):
        edges.append((i, len(vertices) - 1))

    for i in range(0, len(vertices)):
        edges.append((i, len(vertices) - 2))

    vertices.append(base_center)
    vertices.append(apex)
    position = [position[0], position[1], position[2], 1.0]

    Result.MainScene.ObjectsOnScene.append([name, position, radius, height, segments, vertices, edges])

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

    position = [position[0], position[1], position[2], 1.0]

    Result.MainScene.ObjectsOnScene.append([name, position, None, None, None, vertices, edges])
#! ----------

#! Edit mode

#! ----------

#! Character control functions

#! ----------