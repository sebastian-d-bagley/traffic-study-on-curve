import os
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# All the linear algebra helper functions already written
def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

def xy_to_angles(x, y, width, height, horizontal_fov, vertical_fov):
    x_shifted = x - width / 2
    y_shifted = y - height / 2
    x_shifted = -x_shifted
    y_shifted = -y_shifted

    horizontal_angle_deg = (x_shifted / width) * horizontal_fov
    vertical_angle_deg = (y_shifted / height) * vertical_fov

    return math.radians(horizontal_angle_deg), math.radians(vertical_angle_deg)

def unit_vector_from_angles(horizontal_angle, vertical_angle):
    x = math.cos(vertical_angle) * math.cos(horizontal_angle)
    y = math.cos(vertical_angle) * math.sin(horizontal_angle)
    z = math.sin(vertical_angle)
    return (x, y, z)

def scalar_multiply(a, b):
    return [a[i] * b for i in range(len(a))]

def add_vectors(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def plane_intersection(plane_normal, plane_point, ray_direction, ray_point):
    d = -dot(plane_normal, plane_point)
    denom = dot(plane_normal, ray_direction)
    if abs(denom) < 1e-8:
        # Ray is parallel
        return ray_point  # fallback
    t = -(dot(plane_normal, ray_point) + d) / denom
    return add_vectors(ray_point, scalar_multiply(ray_direction, t))

def project_onto_plane(point, plane_origin, normal):
    normal = np.array(normal) / np.linalg.norm(normal)
    point_arr = np.array(point)
    plane_origin_arr = np.array(plane_origin)
    vector_to_point = point_arr - plane_origin_arr
    distance = np.dot(vector_to_point, normal)
    projected_point = point_arr - distance * normal
    return projected_point

def map_3d_to_2d(plane_origin, normal, axis_x, point):
    normal_unit = np.array(normal) / np.linalg.norm(normal)

    # Project axis_x onto the plane
    axis_x_projected = project_onto_plane(axis_x, plane_origin, normal_unit)
    axis_x_projected = axis_x_projected - plane_origin
    axis_x_unit = axis_x_projected / np.linalg.norm(axis_x_projected)

    # Project the point onto the plane
    point_projected = project_onto_plane(point, plane_origin, normal_unit)

    # Y direction is cross(normal, X_unit)
    axis_y = np.cross(normal_unit, axis_x_unit)
    axis_y_unit = axis_y / np.linalg.norm(axis_y)

    relative_position = point_projected - plane_origin
    x_coord = np.dot(relative_position, axis_x_unit)
    y_coord = np.dot(relative_position, axis_y_unit)
    return (x_coord, y_coord)

# Helper functions to smooth a path and to calculate instantaneous speed
def smooth_path(coords, window_size=5):
    if len(coords) < 2:
        return coords

    coords_arr = np.array(coords)
    smoothed = np.zeros_like(coords_arr)
    half_window = window_size // 2

    for i in range(len(coords_arr)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(coords_arr), i + half_window + 1)
        smoothed[i] = coords_arr[start_idx:end_idx].mean(axis=0)

    return smoothed
def compute_instantaneous_speed(positions_2d, time_deltas):
    speeds = [np.nan]  # no speed for the first data point
    for i in range(1, len(positions_2d)):
        dist = np.linalg.norm(np.array(positions_2d[i]) - np.array(positions_2d[i-1]))
        dt = time_deltas[i]
        if dt == 0:
            speeds.append(np.nan)
        else:
            speeds.append(dist / dt)
    return np.array(speeds)

def smooth_speed(speed_array, window=5):
    valid_speed = np.nan_to_num(speed_array, nan=0.0)
    kernel = np.ones(window) / window
    speed_smooth = np.convolve(valid_speed, kernel, mode='same')
    return speed_smooth

def compute_combined_avg_speed_for_video(
    df_video,
    plane_origin,
    normal,
    axis_x,
    width,
    height,
    hfov,
    vfov,
    max_x=132.7,
    max_y=106.2,
    min_points=15,
    max_dist_jump=30.0,
    min_distance=10.0
):

    if len(df_video) < min_points:
        return None

    # Sort by time
    df_video.sort_values(by="frame_index", ascending=True, inplace=True)

    # Convert each detection to 2D point on the plane
    car_positions_2d = []
    time_list = []

    for _, row in df_video.iterrows():
        width = row["Width"]
        height = row["Height"]

        x_pix = row["x"]
        y_pix = row["y"]
        time_sec = row["time_sec"]

        horiz_angle, vert_angle = xy_to_angles(
            x_pix, y_pix, width, height, hfov, vfov
        )
        ray_dir = unit_vector_from_angles(horiz_angle, vert_angle)
        intersection_3d = plane_intersection(normal, plane_origin, ray_dir, (0, 0, 0))
        p2d = map_3d_to_2d(plane_origin, normal, axis_x, intersection_3d)

        # Filter out points if they exceed plane coords
        if p2d[0] > max_x or p2d[1] > max_y:
            continue

        car_positions_2d.append(p2d)
        time_list.append(time_sec)

    if len(car_positions_2d) < min_points:
        return None

    # throw out the whole video if the jump is too big
    for i in range(1, len(car_positions_2d)):
        dist_jump = np.linalg.norm(np.array(car_positions_2d[i]) - np.array(car_positions_2d[i - 1]))
        if dist_jump > max_dist_jump:
            print(f"Skipping video. jump of {dist_jump:.2f} ft")
            return None

    car_2d_arr = np.array(car_positions_2d)
    time_arr = np.array(time_list)

    # Smooth the path
    car_2d_smooth = smooth_path(car_2d_arr, window_size=5)

    # Compute instantaneous speeds
    delta_time = np.diff(time_arr, prepend=time_arr[0])
    speed_smooth_path = compute_instantaneous_speed(car_2d_smooth, delta_time)

    # Convert to mph
    ft_s_to_mph = 3600.0 / 5280.0
    speed_smooth_mph = speed_smooth_path * ft_s_to_mph

    # Smooth the smoothed
    speed_smooth_values_mph = smooth_speed(speed_smooth_mph, window=5)
    avg_smoothed_speed = np.nanmean(speed_smooth_values_mph)

    # Compute overall start to end speed
    total_distance_ft = np.linalg.norm(car_2d_smooth[-1] - car_2d_smooth[0])
    total_time_sec = time_arr[-1] - time_arr[0]
    if total_time_sec <= 0:
        return None

    # IF it doesn't move enough, skip it
    if total_distance_ft < min_distance:
        print(f"Skipping video due to displacement of {total_distance_ft:.2f} ft (below threshold of {min_distance} ft).")
        return None

    start_end_speed_mph = (total_distance_ft / total_time_sec) * ft_s_to_mph

    combined_avg_mph = np.nanmean([avg_smoothed_speed, start_end_speed_mph])

    # By default, return start_end_speed.
    # Only if the combined_avg is less than (start_end_speed + 5) do we average them.
    # This is because at a first approximation the cars will be travelling in a straight line, and this is the most accurate method for measuring speed
    final_speed_mph = start_end_speed_mph
    if combined_avg_mph < (start_end_speed_mph + 5):
        final_speed_mph = 0.5 * (combined_avg_mph + start_end_speed_mph)

    return final_speed_mph

def plot_speed_histogram(
    csv_path,
    plane_origin,
    normal,
    axis_x,
    width,
    height,
    hfov,
    vfov,
    bin_size=5,
    max_speed=100,
    min_points=15,
    max_dist_jump=30.0,
    min_distance=10.0
):
    # Load the whole CSV
    df = pd.read_csv(csv_path)

    # Identify unique video names
    video_names = df["video_filename"].unique()

    # all the final speeds in this list
    speeds = []
    names = []

    skip_count = 0
    # Process each video
    for video_name in video_names:
        df_video = df[df["video_filename"] == video_name].copy()
        final_speed = compute_combined_avg_speed_for_video(
            df_video,
            plane_origin,
            normal,
            axis_x,
            width,
            height,
            hfov,
            vfov,
            max_x=132.7,
            max_y=106.2,
            min_points=min_points,
            max_dist_jump=max_dist_jump,
            min_distance=min_distance
        )
        if final_speed is not None:
            speeds.append(final_speed)
            names.append(video_name)
        else:
            skip_count += 1
            print(f"Skipping '{video_name}' due to insufficient data, excessive jump, or displacement below threshold.")
            print(f"Total skipped so far: {skip_count}")

    # If we have no valid speeds, just exit
    if not speeds:
        print("No valid speeds found.")
        return

    print("\n--------------------------------------\n")
    filtered_speeds = [(s, n) for s, n in zip(speeds, names) if s > 22]
    if filtered_speeds:
        min_val, min_vid = min(filtered_speeds, key=lambda x: x[0])
        print(f"Min speed above 1 mph: {min_val:.2f} mph")
        print(f"Video: {min_vid}")

    # Create the bins
    bins = np.arange(0, max_speed + bin_size, bin_size)
    print(np.percentile(speeds, 85))
    plt.figure(figsize=(9, 6))
    plt.hist(speeds, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel("Speed (mph)")
    plt.ylabel("Number of Cars")
    plt.title("Distribution of Speeds (New Logic)")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


# Graphing the histogram
if __name__ == "__main__":
    plot_speed_histogram(
        csv_path="car_tracking_data_all.csv",
        plane_origin=[0, 0, -17/1.4],
        normal=(-0.23, 0.235, 1.4),
        axis_x=(1, 0, 0),
        width=2560,
        height=1920,
        hfov=80,
        vfov=42,
        bin_size=2,       # each bin covers 2 mph
        max_speed=30,    # histogram up to 100 mph
        min_points=15,    # skip cars with fewer than 15 valid detections
        max_dist_jump=30, # disregard the entire car if any jump exceeds 30 feet
        min_distance=60.0 # only include cars that travel at least 10 feet
    )
