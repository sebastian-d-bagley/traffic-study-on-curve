# This function graphs the car's position on a 2D plane of the road using the CSV file with all the car positions and an image of the road drawn in red.
# It also computes and graphs the car's speed over time.

import os
import cv2
import numpy as np
import pandas as pd
import math
import pytesseract
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# load the classifier
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# helper
def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

# Gets the timestamp from the first frame of the video using tesseract OCR
def extract_timestamp_from_video(
    video_path: str,
    bounding_box: tuple
) -> str:
    x1, y1, x2, y2 = bounding_box

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read first frame")
    cap.release()

    # Crop to region of interest (for OCR)
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    extracted_text = pytesseract.image_to_string(roi_rgb)
    return extracted_text.strip()

def process_video(
    video_path,
    net,
    output_layers,
    classes,
    bounding_box,
    skip_frames_no_det=10,
    skip_frames_det=0,
    frames_after_det=30
):
    # Get the timestamp
    video_timestamp = extract_timestamp_from_video(video_path, bounding_box)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_time = 1.0 / fps

    detections = []
    total_frame_index = 0
    frames_since_detection = 99999
    detection_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate by 180 because the camera is upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Decide if we skip this frame
        current_skip = skip_frames_det if detection_mode else skip_frames_no_det
        if (total_frame_index % (current_skip + 1)) != 0:
            total_frame_index += 1
            continue

        # Set up the frame for detection model
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416),
            (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Track cars only only
                if classes[class_id] == "car" and confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] *
                        np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        time_from_first_frame = total_frame_index * frame_time

        if len(indices) > 0:
            # We have detections
            frames_since_detection = 0
            detection_mode = True
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # "special point" near bottom center
                point_x = x + w // 2
                point_y = y + int(0.75 * h)

                detections.append({
                    "Video": os.path.basename(video_path),
                    "Extracted_Timestamp": video_timestamp,
                    "FrameIndex": total_frame_index,
                    "X": point_x,
                    "Y": point_y,
                    "Time_From_First_Frame": time_from_first_frame
                })
        else:
            frames_since_detection += 1

        if frames_since_detection > frames_after_det:
            detection_mode = False

        total_frame_index += 1

    cap.release()
    return detections

# All helper functions for the 3D to 2D projection
# Better explanations in `car_from_csv_on_road.py`
def xy_to_angles(x, y, width, height, horizontal_fov, vertical_fov):
    x_shifted = x - width / 2
    y_shifted = y - height / 2

    # Invert if you want a particular orientation:
    x_shifted = -x_shifted
    y_shifted = -y_shifted

    horizontal_angle_deg = (x_shifted / width) * horizontal_fov
    vertical_angle_deg   = (y_shifted / height) * vertical_fov

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
        # Ray is parallel or nearly so
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

    axis_x_projected = project_onto_plane(axis_x, plane_origin, normal_unit)
    axis_x_projected = axis_x_projected - plane_origin
    axis_x_unit = axis_x_projected / np.linalg.norm(axis_x_projected)

    point_projected = project_onto_plane(point, plane_origin, normal_unit)

    axis_y = np.cross(normal_unit, axis_x_unit)
    axis_y_unit = axis_y / np.linalg.norm(axis_y)

    relative_position = point_projected - plane_origin
    x_coord = np.dot(relative_position, axis_x_unit)
    y_coord = np.dot(relative_position, axis_y_unit)

    return (x_coord, y_coord)
def order_points(points, max_distance=20, min_point_distance=5):
    tree = KDTree(points)
    ordered_points = []
    remaining_points = set(map(tuple, points))

    while remaining_points:
        segment = []
        current_point = remaining_points.pop()
        segment.append(current_point)

        while True:
            nearest_dist, nearest_idx = tree.query(current_point, k=len(points))
            for idx in nearest_idx:
                candidate = tuple(points[idx])
                if (
                    candidate in remaining_points
                    and np.linalg.norm(np.array(current_point) - np.array(candidate)) < max_distance
                ):
                    # Only add candidate to segment if it's not super close to the last point
                    if (len(segment) == 0
                        or np.linalg.norm(np.array(segment[-1]) - np.array(candidate)) > min_point_distance):
                        segment.append(candidate)
                    remaining_points.remove(candidate)
                    current_point = candidate
                    break
            else:
                break

        ordered_points.append(np.array(segment))

    return ordered_points
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
    speeds = [np.nan]
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

def graph_car_positions(
    csv_path,
    video_name,
    image_path,
    plane_origin,
    normal,
    axis_x,
    width,
    height,
    hfov,
    vfov
):

    # Load the CSV with everything and then sort to only have the video we want
    df = pd.read_csv(csv_path)
    df_video = df[df["video_filename"] == video_name].copy()
    if len(df_video) == 0:
        print(f"No data found for video: {video_name}")
        return

    # Sort to make sure it's in order
    df_video.sort_values(by="frame_index", ascending=True, inplace=True)

    # Load all the red pixels from the annotated image so we have the road outline 
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Find red pixels
        red_pixels = np.argwhere(
            (image[:, :, 0] > 150) &
            (image[:, :, 1] < 100) &
            (image[:, :, 2] < 100)
        )
        points_3d = []
        for (py, px) in red_pixels:
            horiz_angle, vert_angle = xy_to_angles(px*1, py*1, width, height, hfov, vfov)
            ray_dir = unit_vector_from_angles(horiz_angle, vert_angle)
            intersection_3d = plane_intersection(normal, plane_origin, ray_dir, (0, 0, 0))
            points_3d.append(intersection_3d)

        points_2d = [map_3d_to_2d(plane_origin, normal, axis_x, p) for p in points_3d]
        if len(points_2d) > 0:
            ordered_segments = order_points(np.array(points_2d))
            for segment in ordered_segments:
                plt.plot(segment[:, 0], segment[:, 1], c='r')

    car_positions_2d = []
    time_list = []

    # Load the xy from the CSV with the bounding box point saved and then map it to the 2D road plane we establish above 
    for _, row in df_video.iterrows():
        x_pix = row["x"]
        y_pix = row["y"]
        horiz_angle, vert_angle = xy_to_angles(x_pix, y_pix, width, height, hfov, vfov)
        ray_dir = unit_vector_from_angles(horiz_angle, vert_angle)
        intersection_3d = plane_intersection(normal, plane_origin, ray_dir, (0, 0, 0))
        p2d = map_3d_to_2d(plane_origin, normal, axis_x, intersection_3d)

        if True:
            car_positions_2d.append(p2d)
            time_list.append(row["time_sec"])

    car_2d_arr = np.array(car_positions_2d)
    if len(car_2d_arr) == 0:
        print("nothing")
        return

    # Plot unsmoothed car positions
    plt.scatter(car_2d_arr[:, 0], car_2d_arr[:, 1], c='b', label='Car Positions', alpha=0.7)

    # Smooth the car path and then plot it again
    car_2d_smooth = smooth_path(car_2d_arr, window_size=5)
    plt.plot(car_2d_smooth[:, 0], car_2d_smooth[:, 1], c='g', linestyle='--',
             marker='x', label='Smoothed Car Path')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(f'Car Path Projection: {video_name}')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()

    # (F) Compute speed in units/sec
    time_list = np.array(time_list)
    delta_time = np.diff(time_list, prepend=time_list[0])  # first difference is 0

    speed_original = compute_instantaneous_speed(car_2d_arr, delta_time)
    speed_smooth_path = compute_instantaneous_speed(car_2d_smooth, delta_time)

    # (G) Convert ft/s -> mph if your plane is in feet (optional)
    ft_s_to_mph = 3600.0 / 5280.0
    speed_original_mph = speed_original * ft_s_to_mph
    speed_smooth_mph   = speed_smooth_path * ft_s_to_mph

    # Smooth the smoothed speed again for more smoothness
    speed_smooth_values_mph = smooth_speed(speed_smooth_mph, window=5)

    plt.figure(figsize=(10,5))

    # Drop the first 5 and the last 5 to avoid any weird edge effects
    start_slice = 5
    end_slice = -5 if len(speed_original_mph) > 10 else len(speed_original_mph)

    valid_indices = range(start_slice, len(speed_original_mph) + end_slice)

    avg_smoothed_speed = np.nanmean(speed_smooth_values_mph)
    print(f"Average Smoothed Speed: {avg_smoothed_speed:.2f} mph")
    total_distance_ft = np.linalg.norm(car_2d_smooth[-1] - car_2d_smooth[0])
    total_time_sec = time_list[-1] - time_list[0]
    start_end_speed_mph = (total_distance_ft / total_time_sec) * ft_s_to_mph
    print(f"Start-End Speed: {start_end_speed_mph:.2f} mph")
    combined_avg = np.nanmean([avg_smoothed_speed, start_end_speed_mph])
    print(f"Combined Avg: {combined_avg:.2f} mph")

    plt.plot(valid_indices, speed_original_mph[start_slice:end_slice],
             label='Original Speed (mph)', alpha=0.7)
    plt.plot(valid_indices, speed_smooth_mph[start_slice:end_slice],
             label='Speed (Smoothed Path) (mph)', alpha=0.7)
    plt.plot(valid_indices, speed_smooth_values_mph[start_slice:end_slice],
             label='Speed (Path + Speed Smoothed)', linestyle='--', linewidth=2)

    plt.title(f'Speed vs. Frame Index: {video_name}')
    plt.xlabel('Frame Index (cropped)')
    plt.ylabel('Speed (mph)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    graph_car_positions(
        csv_path="car_tracking_data_all.csv",
        video_name="RecS04_20250224_100803_100829_391480800_12DB65.mp4",
        image_path="roads_drawn.png",
        plane_origin=[0, 0, -17/1.4],
        normal=(-0.23, 0.235, 1.4),
        axis_x=(1, 0, 0),
        width=640,
        height=480,
        hfov=80,
        vfov=42
    )
