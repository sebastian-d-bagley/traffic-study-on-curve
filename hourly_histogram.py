import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the CSV file
df = pd.read_csv("car_tracking_data_all.csv")

# Extract date/time from each video filename
def extract_datetime(filename):
    parts = filename.split('_')
    date_str = parts[1]
    time_str = parts[2]
    datetime_str = f"{date_str} {time_str}"
    return datetime.strptime(datetime_str, "%Y%m%d %H%M%S")

# Add column with the datetime objects
df['datetime'] = df['video_filename'].apply(extract_datetime)

# The CSV contains multiple rows per video (each representing a car location in a frame).
# Sort by video filename and ensure that there are more than 5 frames per video
df_valid = df.groupby('video_filename').filter(lambda g: len(g) >= 5)

# Remove duplicates to have only one row per video (all we need is the fact of the car for this graphing)
unique_videos = df_valid.drop_duplicates(subset=['video_filename'])

# Create columns for date and hour-of-day
unique_videos['date'] = unique_videos['datetime'].dt.date
unique_videos['hour_of_day'] = unique_videos['datetime'].dt.hour

# These are the days that we want to graph
days_to_graph = [
    "2025-02-18",
    "2025-02-19",
    "2025-02-20",
    "2025-02-21",
    "2025-02-22",
    "2025-02-23",
]

# Convert day strings to date objects
desired_dates = set(datetime.strptime(day, "%Y-%m-%d").date() for day in days_to_graph)

# Filter to only those days
unique_videos = unique_videos[unique_videos['date'].isin(desired_dates)]

# Count how many cars per day
cars_per_day_hour = (
    unique_videos
    .groupby(['date', 'hour_of_day'])
    .size()
    .reset_index(name='count')
)

# Average the counts across all selected days
avg_cars_by_hour = cars_per_day_hour.groupby('hour_of_day')['count'].mean()

# Make sure there are 24 hours
all_hours = range(24)
avg_cars_by_hour = avg_cars_by_hour.reindex(all_hours, fill_value=0)

# Plot
plt.figure(figsize=(10, 6))
avg_cars_by_hour.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xticks(range(24), range(24))  # Label x-axis as 0..23
plt.xlabel('Hour of Day')
plt.ylabel('Avg # of Cars (1 Car = 1 Video)') 
plt.title(f'Average Cars per Hour (Across {len(days_to_graph)} Days)')
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_cars_per_hour.png")
plt.show()

# Calculate total
total_cars_per_day = unique_videos.groupby('date').size()

# Calculate average
average_cars_per_day = total_cars_per_day.mean()

print(f'Average number of cars per day: {average_cars_per_day:.2f}')