from collections import defaultdict

import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11x.pt")

# Open the video file
video_path = "tests/CARROS.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start_time = time.time()
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        if(results != None):
            # Get the boxes and track IDs
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Your code for processing track_ids

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                
                end_time = time.time()
                fps = 1/ (end_time - start_time)
                framefps = "FPS: {:.2f}".format(fps)
                cv2.rectangle(annotated_frame, (10,1), (120,20), (0, 0,0), -1)
                cv2.putText(annotated_frame, framefps, (15,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", annotated_frame)
            else:
                print("No tracks found in this frame")
                 # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()