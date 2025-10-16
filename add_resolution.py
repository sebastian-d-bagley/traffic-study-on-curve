import cv2
import pandas as pd

# This script reads a CSV file containing video file paths,
# extracts the resolution and framerate of each video,
# and saves the information back to the CSV file.

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def update_csv_with_video_info(csv_path, video_column, output_csv):
    df = pd.read_csv(csv_path)

    # Create new columns
    df["Width"] = None
    df["Height"] = None
    df["Framerate"] = None
    i = 0
    past = ""
    video_path = ""
    # Process each video file
    for index, row in df.iterrows():
        past = video_path
        video_path = str(row[video_column])
        if past == video_path:
            df.at[index, "Width"] = width
            df.at[index, "Height"] = height
            df.at[index, "Framerate"] = fps
            i +=1 
            print(i)
            continue

        width, height, fps = get_video_properties("full_footage/" + video_path)
        df.at[index, "Width"] = width
        df.at[index, "Height"] = height
        df.at[index, "Framerate"] = fps
        i+=1

    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved as {output_csv}")

csv_path = "car_tracking_data_all.csv"  # Input CSV file
video_column = "video_filename"  # Column name containing video paths
output_csv = "car_tracking_data_all.csv"  # Output CSV file

update_csv_with_video_info(csv_path, video_column, output_csv)
