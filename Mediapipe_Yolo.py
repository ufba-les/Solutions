import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

def process_video_with_pose_detection_inside_bed(video_path, model_path, skip_frames=None):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        exit()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    # Flag to indicate if bed bounding box has been obtained
    bed_bounding_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_frames is not None and current_frame % (skip_frames + 1) != 0:
            current_frame += 1
            continue

        # If bed bounding box is not yet obtained, run YOLO on this frame
        if bed_bounding_box is None:
            # Run inference on the frame
            results = model(frame)
            result = results[0]

            # Get class names
            class_names = model.names

            # Define the class name for the bed
            bed_class_name = 'hospital-beds'  # Use the exact class name from your model

            # Access detections
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Class ID and name
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id]

                    if class_name == bed_class_name:
                        # Bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Ensure coordinates are within image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        # Store the bounding box coordinates
                        bed_bounding_box = (x1, y1, x2, y2)
                        print(f"Bed detected with bounding box: {bed_bounding_box}")
                        break  # Use only the first detected bed

                if bed_bounding_box is None:
                    print("Warning: No bed detected in the initial frame.")
                    break  # Exit if no bed is detected
            else:
                print("Warning: No detections in the initial frame.")
                break  # Exit if no detections

        # Use the obtained bounding box to crop the frame
        x1, y1, x2, y2 = bed_bounding_box

        # Crop the frame to the bed bounding box
        cropped_frame = frame[y1:y2, x1:x2]

        # Check if the cropped frame is of sufficient size for MediaPipe
        min_width, min_height = 100, 100  # Minimum dimensions for reliable detection
        if cropped_frame.shape[1] < min_width or cropped_frame.shape[0] < min_height:
            print("Cropped frame is too small for pose detection.")
            current_frame += 1
            continue

        # Convert the cropped frame to RGB
        cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # Process the cropped frame with MediaPipe Pose
        result = pose.process(cropped_frame_rgb)

        # Draw the pose landmarks on the cropped frame
        if result.pose_landmarks:
            # Get the image dimensions
            height, width, _ = cropped_frame.shape

            # Get torso landmarks: shoulders and hips
            torso_landmarks_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                       mp_pose.PoseLandmark.LEFT_HIP.value,
                                       mp_pose.PoseLandmark.RIGHT_HIP.value]

            torso_outside_bed = False

            for idx in torso_landmarks_indices:
                landmark = result.pose_landmarks.landmark[idx]
                # Convert normalized coordinates to pixel coordinates in the cropped frame
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Map the coordinates back to the original frame
                x_global = x + x1
                y_global = y + y1

                # Check if the landmark is outside the bed bounding box in the original frame
                if not (x1 <= x_global <= x2 and y1 <= y_global <= y2):
                    torso_outside_bed = True
                    tempo_atual = current_frame / frame_rate
                    warning_text = f"Warning: Torso outside bed at {tempo_atual:.2f}s"
                    print(warning_text)
                    # Overlay warning on the original frame
                    cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    break  # No need to check other landmarks

                # Optionally, draw the landmark position on the original frame
                cv2.circle(frame, (x_global, y_global), 5, (255, 0, 0), -1)

            # Draw the pose landmarks on the cropped frame (optional)
            mp_drawing.draw_landmarks(cropped_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Optionally, draw the bed bounding box on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the original frame with overlays
        cv2.imshow('Video with Pose Detection', frame)

        # Display the cropped frame with pose landmarks (optional)
        # cv2.imshow('Cropped Frame', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'X:/Videos_Hospital/WIN_20240619_16_27_45_Pro.mp4'  # Update this path
    model_path = 'X:/best.pt'  # Update this path
    skip_frames = 1  # Adjust as needed

    process_video_with_pose_detection_inside_bed(video_path, model_path, skip_frames)
