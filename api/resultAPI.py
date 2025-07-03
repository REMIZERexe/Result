import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import cupy as cp
from PIL import Image

#! Settings by default
fullscreen = True
camera_position = [0, 0, -5]
yaw = 0 # Horizontal camera rotation
pitch = 0 # Vertical camera rotation
FOV = 400
objects_onscene = []
engine_window = None

def set_window_instance(win):
    global engine_window
    engine_window = win
#! ----------

#! Rendering functions
def render_object(obj):
    from api.app.init import centerX, centerY
    name, position, sizeX, sizeY, sizeZ, verticies, edges = obj
    shifted = cp.array(verticies) + cp.array(position)
    points = project_points(shifted, camera_position, FOV, centerX, centerY)
    points = cp.asnumpy(points)

    for edge in edges:
        engine_window.draw_line(points[edge[0]], points[edge[1]])

def project_points(points, cam_pos, fov, centerx, centery):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x -= cam_pos[0]
    y -= cam_pos[1]
    z -= cam_pos[2]

    cos_yaw = cp.cos(cp.radians(yaw))
    sin_yaw = cp.sin(cp.radians(yaw))
    cos_pitch = cp.cos(cp.radians(pitch))
    sin_pitch = cp.sin(cp.radians(pitch))

    rotatedX = x * cos_yaw - z * sin_yaw
    rotatedZ = x * sin_yaw + z * cos_yaw

    rotatedY = y * cos_pitch - rotatedZ * sin_pitch
    rotatedZ = y * sin_pitch + rotatedZ * cos_pitch

    inv_z = 1.0 / rotatedZ
    x_screen = (rotatedX * fov * inv_z + centerx).astype(cp.int32)
    y_screen = (rotatedY * fov * inv_z + centery).astype(cp.int32)

    return cp.stack((x_screen, y_screen), axis=1)

def rotate_object(axis: str, object_name: str, angle: float):
    for i, obj in enumerate(objects_onscene):
        if obj[0] != object_name:
            print("rotate_object(" + axis + ", " + object_name + ", " + str(angle) + "): " + "The entered object does not exist!")
            return
        if axis.upper() != "X" and axis.upper() != "Y" and axis.upper() != "Z":
            print("rotate_object(" + axis + ", " + object_name + ", " + str(angle) + "): " + "You can't rotate an object on a non-existent axis!")
            return
                    
        vertices = obj[5]
        rotated = []

        for vert in vertices:
            x, y, z = vert
            match axis.upper():
                case "X":
                    cos_theta = math.cos(math.radians(angle))
                    sin_theta = math.sin(math.radians(angle))
                    y_rotated = y * cos_theta - z * sin_theta
                    z_rotated = y * sin_theta + z * cos_theta
                    rotated.append((x, y_rotated, z_rotated))

                case "Y":
                    cos_theta = math.cos(math.radians(angle))
                    sin_theta = math.sin(math.radians(angle))
                    x_rotated = x * cos_theta - z * sin_theta
                    z_rotated = x * sin_theta + z * cos_theta
                    rotated.append((x_rotated, y, z_rotated))

                case "Z":
                    cos_theta = math.cos(math.radians(angle))
                    sin_theta = math.sin(math.radians(angle))
                    x_rotated = x * cos_theta - y * sin_theta
                    y_rotated = x * sin_theta + y * cos_theta
                    rotated.append((x_rotated, y_rotated, z))

        objects_onscene[i] = (obj[0], obj[1], obj[2], obj[3], obj[4], rotated, obj[6])
        return

def texture_calculation(image_path, screen_width, screen_height, point1, point2, point3, point4):
    pil_image = Image.open(image_path).convert("RGBA")

    image = cp.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    height, width = image.shape[:2]

    src_verticies = cp.float32([
        [0, 0], [1023, 0], [width, height], [0, height]
    ])
    dst_verticies = cp.float32([
        point1, point2, point3, point4
    ])

    # Матрица преобразования и применение
    matrix = cv2.getPerspectiveTransform(src_verticies, dst_verticies)
    transformed_image = cv2.warpPerspective(image, matrix, (screen_width, screen_height), flags=cv2.INTER_LINEAR)

    # Преобразование обратно в Pillow
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGRA2RGBA)
    pil_transformed = Image.fromarray(transformed_image)

    return pil_transformed

#! Create objects
def create_instance(obj_name: str, inst_name: str, position: tuple):
    if len(position) > 3 or len(position) < 3:
        print("create_instance(" + obj_name + ", " + inst_name + ", " + position + "): " + "The entered position is invalid!")

    for obj in objects_onscene:
        if obj[0] == obj_name:
            objects_onscene.append((inst_name, position, obj[2], obj[3], obj[4], obj[5], obj[6]))

def create_cube(position, name, sizeX, sizeY, sizeZ):
    offset_x = sizeX / 2
    offset_y = sizeY / 2
    offset_z = sizeZ / 2

    verticies = cp.array([
        ( offset_x,  offset_y,  offset_z),
        ( offset_x, -offset_y,  offset_z),
        (-offset_x, -offset_y,  offset_z),
        (-offset_x,  offset_y,  offset_z),
        ( offset_x,  offset_y, -offset_z),
        ( offset_x, -offset_y, -offset_z),
        (-offset_x, -offset_y, -offset_z),
        (-offset_x,  offset_y, -offset_z),
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    objects_onscene.append((name, position, sizeX, sizeY, sizeZ, verticies, edges))

def create_sphere(position, name, radius, segments, rings):
    verticies = []
    edges = []

    for i in range(rings):
        theta = math.pi * i / (rings - 1)
        for j in range(segments):
            phi = 2 * math.pi * j / segments

            x = radius * math.cos(phi) * math.sin(theta)
            z = radius * math.sin(phi) * math.sin(theta)
            y = radius * math.cos(theta)

            verticies.append((x, y, z))
            cp.array(verticies)
    
    for i in range(rings):
        for j in range(segments):
            current = i * segments + j
            right = i * segments + (j + 1) % segments
            if j < segments - 1 or True:
                edges.append((current, right))

            if i < rings - 1:
                below = (i + 1) * segments + j
                edges.append((current, below))
                cp.array(edges)

    objects_onscene.append((name, position, radius, segments, rings, verticies, edges))

def load_model(directory: str, position: tuple, name: str):
    import re
    verticies = []
    edges = []

    with open(directory, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                vert_raw = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
                vert = [float(x) for x in vert_raw]
                verticies.append(tuple(vert))
            if line.startswith("f "):
                parts = line.strip().split()[1:]
                vertex_indices = [int(p.split('/')[0]) for p in parts]
                vertex_indices = [i - 1 for i in vertex_indices]
                pair_edge = [(vertex_indices[i], vertex_indices[i+1]) for i in range(0, len(vertex_indices), 2)]
                edges.append(pair_edge[0])
                edges.append(pair_edge[1])
        
    verticies = cp.asnumpy(verticies)

    objects_onscene.append((name, position, None, None, None, verticies, edges))
#! ----------

#! Edit mode

#! ----------

#! Character control functions
# def handle_camera_movement():
#     global keys
#     keys = pygame.key.get_pressed()
#     move_speed = 1
#     movement = [0, 0, 0]

#     if keys[pygame.K_z]:  # Forward
#         camera_position[2] += move_speed * math.cos(math.radians(yaw))
#         camera_position[0] += move_speed * math.sin(math.radians(yaw))
#         movement[2] += move_speed * math.cos(math.radians(yaw))
#         movement[0] += move_speed * math.sin(math.radians(yaw))
#     if keys[pygame.K_s]:  # Backward
#         camera_position[2] -= move_speed * math.cos(math.radians(yaw))
#         camera_position[0] -= move_speed * math.sin(math.radians(yaw))
#         movement[2] -= move_speed * math.cos(math.radians(yaw))
#         movement[0] -= move_speed * math.sin(math.radians(yaw))
#     if keys[pygame.K_q]:  # Left
#         camera_position[0] -= move_speed * math.cos(math.radians(yaw))
#         camera_position[2] += move_speed * math.sin(math.radians(yaw))
#         movement[0] -= move_speed * math.cos(math.radians(yaw))
#         movement[2] += move_speed * math.sin(math.radians(yaw))
#     if keys[pygame.K_d]:  # Right
#         camera_position[0] += move_speed * math.cos(math.radians(yaw))
#         camera_position[2] -= move_speed * math.sin(math.radians(yaw))
#         movement[0] += move_speed * math.cos(math.radians(yaw))
#         movement[2] -= move_speed * math.sin(math.radians(yaw))
#     if keys[pygame.K_SPACE]:  # Up
#         camera_position[1] -= move_speed
#         movement[1] -= move_speed
#     if keys[pygame.K_LSHIFT]:  # Down
#         camera_position[1] += move_speed
#         movement[1] += move_speed

#     length = math.sqrt(movement[0]**2 + movement[1]**2 + movement[2]**2)
#     if length > 0:
#         movement[0] /= length
#         movement[1] /= length
#         movement[2] /= length

#     camera_position[0] += movement[0] * move_speed
#     camera_position[1] += movement[1] * move_speed
#     camera_position[2] += movement[2] * move_speed

def handle_camera_rotation():
    from api.app.init import dx, dy
    global yaw, pitch

    sensitivity = 0.15
    yaw += dx * sensitivity
    pitch += dy * sensitivity

    pitch = max(-89, min(89, pitch))
#! ----------