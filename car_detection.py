# Good for getting an idea of how the car detection is working and what will actually be saved to the CSV

import cv2
import numpy as np
import time
import pandas as pd

# Load YOLO car model
net = cv2.dnn.readNet(r"Model Weights\yolov4.weights", r"Model Weights\yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(r"Model Weights\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video file (this one is just an example)
video_path = r"full_footage\RecS04_20250224_123830_123859_391480800_1408AC.mp4"
cap = cv2.VideoCapture(video_path)

# Get frame rate to calculate time difference
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps

# Store car position and delta time
car_points = []
prev_time = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame 180 degrees (the camera is set up upside down)
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If detected object is a car and confidence is high enough
            if classes[class_id] == "car" and confidence > 0.8:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)

    # Track time and append detected points
    current_time = prev_time + frame_time  # Increment time
    if len(indices) > 0:  # Ensure we've detected something
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Car {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate the special point (this is where we consider the bottom of the car to be. Imperfect but good enough here for precision) (centered horizontally, lower quarter of bounding box)
            point_x = x + w // 2
            point_y = y + int(0.75 * h)

            # Draw the point
            cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), -1)

            # Append (x, y, time passed since last frame) to list
            time_delta = frame_time if car_points else 0
            car_points.append((point_x, point_y, time_delta))

    prev_time = current_time  # Update time tracker

    # Display the frame
    cv2.imshow("Car Detection Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save car points to CSV
df = pd.DataFrame(car_points, columns=["X", "Y", "Time_Delta"])
df.to_csv("car_tracking_data.csv", index=False)

# Release video capture
cap.release()
cv2.destroyAllWindows()

print("Tracking data saved to car_tracking_data.csv")
