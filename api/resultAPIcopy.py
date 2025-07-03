import pygame
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from PIL import Image

#! Settings by default
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
fullscreen = True
pygame.display.set_caption("Result [pre-alpha]")
camera_position = [0, 0, -400]
yaw = 0 # Horizontal camera rotation
pitch = 0 # Vertical camera rotation
fov = 400
objects_onscene = []
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
#! ----------

#! Main variables
def update_screen_dimensions():
    global width
    global height
    global centerX
    global centerY
    global center
    width, height = screen.get_width(), screen.get_height()
    centerX, centerY = width // 2, height // 2
    center = [centerX, centerY]
update_screen_dimensions()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHTBLUE = (135, 206, 235)
#! ----------

#! Rendering functions
def render_object(obj):
    name, position, sizeX, sizeY, sizeZ, vertices, edges = obj
    points = []
    for vert in vertices:
        points.append(project_point(vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]))

    for edge in edges:
        draw_line(points[edge[0]], points[edge[1]])

def first_person_rotation(x, y, z):
    global yaw, pitch

    cos_yaw = math.cos(math.radians(yaw))
    sin_yaw = math.sin(math.radians(yaw))
    cos_pitch = math.cos(math.radians(pitch))
    sin_pitch = math.sin(math.radians(pitch))

    xz_rotatedX = x * cos_yaw - z * sin_yaw
    xz_rotatedZ = x * sin_yaw + z * cos_yaw

    yz_rotatedY = y * cos_pitch - xz_rotatedZ * sin_pitch
    yz_rotatedZ = y * sin_pitch + xz_rotatedZ * cos_pitch

    return xz_rotatedX, yz_rotatedY, yz_rotatedZ

def project_point(x, y, z):
    x -= camera_position[0]
    y -= camera_position[1]
    z -= camera_position[2]

    x, y, z = first_person_rotation(x, y, z)

    x_screen = int(x * (fov / z) + centerX)
    y_screen = int(y * (fov / z) + centerY)
    return (x_screen, y_screen)

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

    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    height, width = image.shape[:2]

    src_verticies = np.float32([
        [0, 0], [1023, 0], [width, height], [0, height]
    ])
    dst_verticies = np.float32([
        point1, point2, point3, point4
    ])

    # Матрица преобразования и применение
    matrix = cv2.getPerspectiveTransform(src_verticies, dst_verticies)
    transformed_image = cv2.warpPerspective(image, matrix, (screen_width, screen_height), flags=cv2.INTER_LINEAR)

    # Преобразование обратно в Pillow
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGRA2RGBA)
    pil_transformed = Image.fromarray(transformed_image)

    return pil_transformed

def draw_line(start, end):
    if start and end:
        pygame.draw.line(screen, BLACK, start, end, 1)

#! Create objects
def create_instance(obj_name: str, inst_name: str, position: tuple):
    if len(position) > 3:
        print("create_instance(" + obj_name + ", " + inst_name + ", " + position + "): " + "The entered position is invalid!")

    for i, obj in enumerate(objects_onscene):
        if obj[0] == obj_name:
            objects_onscene.append((inst_name, obj[1], obj[2], obj[3], obj[4], obj[5], obj[6]))

def create_cube(position, name, sizeX, sizeY, sizeZ):
    offset_x = sizeX / 2
    offset_y = sizeY / 2
    offset_z = sizeZ / 2

    vertices = [
        ( offset_x,  offset_y,  offset_z),
        ( offset_x, -offset_y,  offset_z),
        (-offset_x, -offset_y,  offset_z),
        (-offset_x,  offset_y,  offset_z),
        ( offset_x,  offset_y, -offset_z),
        ( offset_x, -offset_y, -offset_z),
        (-offset_x, -offset_y, -offset_z),
        (-offset_x,  offset_y, -offset_z),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    objects_onscene.append((name, position, sizeX, sizeY, sizeZ, vertices, edges))

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
    
    for i in range(rings):
        for j in range(segments):
            current = i * segments + j
            right = i * segments + (j + 1) % segments
            if j < segments - 1 or True:
                edges.append((current, right))

            if i < rings - 1:
                below = (i + 1) * segments + j
                edges.append((current, below))

    objects_onscene.append((name, position, radius, segments, rings, verticies, edges))
#! ----------

#! Edit mode

#! ----------

#! Character control functions
def handle_camera_movement():
    global keys
    keys = pygame.key.get_pressed()
    move_speed = 1
    movement = [0, 0, 0]

    if keys[pygame.K_z]:  # Forward
        camera_position[2] += move_speed * math.cos(math.radians(yaw))
        camera_position[0] += move_speed * math.sin(math.radians(yaw))
        movement[2] += move_speed * math.cos(math.radians(yaw))
        movement[0] += move_speed * math.sin(math.radians(yaw))
    if keys[pygame.K_s]:  # Backward
        camera_position[2] -= move_speed * math.cos(math.radians(yaw))
        camera_position[0] -= move_speed * math.sin(math.radians(yaw))
        movement[2] -= move_speed * math.cos(math.radians(yaw))
        movement[0] -= move_speed * math.sin(math.radians(yaw))
    if keys[pygame.K_q]:  # Left
        camera_position[0] -= move_speed * math.cos(math.radians(yaw))
        camera_position[2] += move_speed * math.sin(math.radians(yaw))
        movement[0] -= move_speed * math.cos(math.radians(yaw))
        movement[2] += move_speed * math.sin(math.radians(yaw))
    if keys[pygame.K_d]:  # Right
        camera_position[0] += move_speed * math.cos(math.radians(yaw))
        camera_position[2] -= move_speed * math.sin(math.radians(yaw))
        movement[0] += move_speed * math.cos(math.radians(yaw))
        movement[2] -= move_speed * math.sin(math.radians(yaw))
    if keys[pygame.K_SPACE]:  # Up
        camera_position[1] -= move_speed
        movement[1] -= move_speed
    if keys[pygame.K_LSHIFT]:  # Down
        camera_position[1] += move_speed
        movement[1] += move_speed

    length = math.sqrt(movement[0]**2 + movement[1]**2 + movement[2]**2)
    if length > 0:
        movement[0] /= length
        movement[1] /= length
        movement[2] /= length

    camera_position[0] += movement[0] * move_speed
    camera_position[1] += movement[1] * move_speed
    camera_position[2] += movement[2] * move_speed

def handle_camera_rotation():
    global yaw, pitch
    mouse_dx, mouse_dy = pygame.mouse.get_rel()

    sensitivity = 0.15
    yaw += mouse_dx * sensitivity
    pitch += mouse_dy * sensitivity

    pitch = max(-89, min(89, pitch))
#! ----------

pp = [(3.67394039744206e-15, -30.0, 0.0),
      (3.181725716174721e-15, -30.0, 1.8369701987210296e-15),
      (1.8369701987210304e-15, -30.0, 3.1817257161747205e-15),
      (2.2496396739927866e-31, -30.0, 3.67394039744206e-15),
      (-1.8369701987210288e-15, -30.0, 3.181725716174721e-15),
      (-3.181725716174721e-15, -30.0, 1.8369701987210296e-15),
      (-3.67394039744206e-15, -30.0, 4.499279347985573e-31),
      (-3.1817257161747213e-15, -30.0, -1.8369701987210288e-15),
      (-1.8369701987210316e-15, -30.0, -3.1817257161747197e-15),
      (-6.748919021978359e-31, -30.0, -3.67394039744206e-15),
      (1.8369701987210304e-15, -30.0, -3.1817257161747205e-15),
      (3.1817257161747197e-15, -30.0, -1.8369701987210316e-15),
      (3.67394039744206e-15, -30.0, -8.998558695971147e-31)]