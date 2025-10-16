import numpy as np
import pandas as pd
import math

# all the linear algebra
# this function just maps the x,y pixels from the camera to the mapped x,y of the road.

def xy_to_angles(x, y, width, height, horizontal_fov, vertical_fov):
    x -= width / 2
    y -= height / 2
    
    x = -x
    y = -y

    horizontal_angle = x / width * horizontal_fov
    vertical_angle = y / height * vertical_fov

    return math.radians(horizontal_angle), math.radians(vertical_angle)

def unit_vector_from_angles(horizontal_angle, vertical_angle):
    x = math.cos(vertical_angle) * math.cos(horizontal_angle)
    y = math.cos(vertical_angle) * math.sin(horizontal_angle)
    z = math.sin(vertical_angle)
    return np.array([x, y, z])

def dot(a, b):
    return np.dot(a, b)

def scalar_multiply(a, b):
    return a * b

def add_vectors(a, b):
    return a + b

def plane_intersection(plane_normal, plane_point, ray_direction, ray_point):
    d = -dot(plane_normal, plane_point)
    t = -(dot(plane_normal, ray_point) + d) / dot(plane_normal, ray_direction)
    return add_vectors(ray_point, scalar_multiply(ray_direction, t))

def project_onto_plane(point, plane_origin, normal):
    normal = normal / np.linalg.norm(normal)
    vector_to_point = point - plane_origin
    distance = np.dot(vector_to_point, normal)
    projected_point = point - distance * normal
    return projected_point

def map_3d_to_2d(plane_origin, normal, axis_x, point):
    normal = normal / np.linalg.norm(normal)
    axis_x = project_onto_plane(axis_x, plane_origin, normal) - plane_origin
    axis_x = axis_x / np.linalg.norm(axis_x)
    point = project_onto_plane(point, plane_origin, normal)
    axis_y = np.cross(normal, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    relative_position = point - plane_origin
    x_coord = np.dot(relative_position, axis_x)
    y_coord = np.dot(relative_position, axis_y)
    return x_coord, y_coord

def process_csv(csv_path, output_csv, plane_origin, normal, axis_x, width, height, hfov, vfov):
    df = pd.read_csv(csv_path)
    
    projected_points = []
    for _, row in df.iterrows():
        ray_direction = unit_vector_from_angles(*xy_to_angles(row['x'], row['y'], width, height, hfov, vfov))
        point_3d = plane_intersection(normal, plane_origin, ray_direction, np.array([0, 0, 0]))
        point_2d = map_3d_to_2d(plane_origin, normal, axis_x, point_3d)
        projected_points.append(point_2d)
    
    df[['Projected_X', 'Projected_Y']] = projected_points
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to {output_csv}")

plane_origin = np.array([0, 0, -18.3252/1.35])
normal = np.array((-0.2, 0.2, 1.35))
axis_x = np.array([1, 0, 0])
process_csv("car_tracking_data.csv", "car_positions_projected.csv", plane_origin, normal, axis_x, 640, 480, 80, 42)
