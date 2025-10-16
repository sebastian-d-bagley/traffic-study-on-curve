# This script processes all the video files in the folder, detects cars using the YOLO detection model, and saves the detected car positions to a CSV file.
# This is the first step once all video files are in the main data folder.

import os
import glob
import cv2
import numpy as np
import pandas as pd

def process_videos_in_folder(folder_path, output_csv):
    # Load the YOLO model
    weights_path = r"Model Weights\yolov4.weights"
    config_path  = r"Model Weights\yolov4.cfg"
    names_path   = r"Model Weights\coco.names"
    
    net = cv2.dnn.readNet(weights_path, config_path)

    # Load class names
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    all_detections = []

    # Go through every video file
    m=0
    for video_file in video_files:
        m+=1
        print("Processing video", m, "of", len(video_files), ":", video_file)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"COuld not open {video_file}. Skipping.")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_index = 0

        # Before we see a car, we should only check around 1 frame per second so the code doesn't take forever
        skip_when_no_car = int(round(fps))
        current_skip = 1
        
        while True:
            # Read one frame
            ret, frame = cap.read()
            if not ret:
                break  # no more frames
            
            # Rotate frame because camera is upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(
                frame, 
                scalefactor=0.00392, 
                size=(416, 416), 
                mean=(0, 0, 0),
                swapRB=True, 
                crop=False
            )
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if classes[class_id] == "car" and confidence > 0.5:
                        center_x, center_y, w, h = (
                            detection[0:4] * np.array([width, height, width, height])
                        ).astype("int")
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Save the position of the car to the boxes list
                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Check if we actually detected any cars
            car_detected = (len(indices) > 0)

            # If a car is detected, lookg through every frame
            if car_detected:
                current_skip = 1

                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    
                    # Old drawing code not really necessary
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Car {confidences[i]:.2f}", 
                                (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 3 quarters of the way down the bounding box
                    point_x = x + w // 2
                    point_y = y + int(0.75 * h)
                    
                    # again old drawing
                    cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), -1)
                    
                    # record everything to the big list of everything
                    time_sec = frame_index / fps
                    all_detections.append((os.path.basename(video_file),
                                           frame_index,
                                           time_sec,
                                           point_x,
                                           point_y))
            else:
                current_skip = skip_when_no_car

            #cv2.imshow("Car Detection", frame)
            
            # Skip frames
            frames_to_skip = current_skip - 1
            for _ in range(frames_to_skip):
                ret_skip, _ = cap.read()
                if not ret_skip:
                    break
                frame_index += 1
            
            frame_index += 1
        
        cap.release()
    
    cv2.destroyAllWindows()

    # Save everything to the CSV
    if all_detections:
        df = pd.DataFrame(all_detections, 
                          columns=["video_filename", "frame_index", "time_sec", "x", "y"])
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No car detections found in any videos. CSV not created or is empty.")


if __name__ == "__main__":
    folder_with_videos = r"Full Footage"
    output_csv_path = "Intermediate CSVs/car_tracking_data.csv"
    process_videos_in_folder(folder_with_videos, output_csv=output_csv_path)



