import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Function to calculate the Signed Distance Function (SDF) from a point to a rectangle
def point_to_rect_sdf(x, y, x1, y1, x2, y2):
    if x1 <= x <= x2 and y1 <= y <= y2:
        # Point is inside the rectangle
        sdf = -min(x - x1, x2 - x, y - y1, y2 - y)
    else:
        # Point is outside the rectangle
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        sdf = np.hypot(dx, dy)
    return sdf

# Function to find the closest point on the rectangle to a given point
def closest_point_on_rect(x, y, x1, y1, x2, y2):
    closest_x = min(max(x, x1), x2)
    closest_y = min(max(y, y1), y2)
    return closest_x, closest_y

# Function to calculate the average displacement of landmarks between frames
def calculate_landmark_displacement(landmarks1, landmarks2):
    if not landmarks1 or not landmarks2:
        return float('inf')  # If landmarks are missing, return infinite displacement
    displacements = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        dx = lm1[0] - lm2[0]
        dy = lm1[1] - lm2[1]
        displacement = np.hypot(dx, dy)
        displacements.append(displacement)
    return np.mean(displacements)

def process_video_with_pose_detection_inside_bed(video_path, model_path, skip_frames=None):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    # Variables to store the bed's bounding box coordinates and previous landmarks
    bed_bounding_box = None
    previous_landmarks = None

    # Threshold for displacement detection between frames
    displacement_threshold = 200  # Adjust based on your video's resolution and desired sensitivity

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is returned

        # Skip frames if skip_frames is specified
        if skip_frames is not None and current_frame % (skip_frames + 1) != 0:
            current_frame += 1
            continue

        # If bed bounding box is not yet obtained, run YOLO on this frame
        if bed_bounding_box is None:
            # Run inference on the frame using YOLOv8 model
            results = model(frame)
            result = results[0]

            # Get class names from the model
            class_names = model.names

            # Define the class name for the bed (ensure it matches your model's class name)
            bed_class_name = 'hospital-beds'  # Update as per your model's class name

            # Access detections
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get class ID and name
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id]

                    if class_name == bed_class_name:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Ensure coordinates are within image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        # Store the bounding box coordinates
                        bed_bounding_box = (x1, y1, x2, y2)
                        print(f"Bed detected with bounding box: {bed_bounding_box}")

                        # Initial person detection within the bed area
                        previous_landmarks = detect_person_in_bed_area(frame, bed_bounding_box, pose)

                        if previous_landmarks is None:
                            print("No pose detected in the bed area.")
                            bed_bounding_box = None  # Reset to trigger bed detection again
                        break  # Use only the first detected bed
            else:
                print("Warning: No detections in the initial frame.")
                break  # Exit if no detections

        else:
            # After initial detection, process the full frame
            # Convert the frame to RGB as required by MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the full frame with MediaPipe Pose
            result = pose.process(frame_rgb)

            # Initialize a list to store SDF values
            sdf_values = []

            # Check if pose landmarks are detected
            if result.pose_landmarks:
                # Get the image dimensions
                height, width, _ = frame.shape

                # Map landmarks to the original frame
                current_landmarks = []
                for landmark in result.pose_landmarks.landmark:
                    x = landmark.x * width
                    y = landmark.y * height
                    current_landmarks.append((x, y))

                # Calculate the displacement between the current and previous landmarks
                displacement = calculate_landmark_displacement(previous_landmarks, current_landmarks)

                # If displacement exceeds threshold, re-detect the person in the bed area
                if displacement > displacement_threshold:
                    print("Significant displacement detected between frames. Re-initializing detection in bed area.")

                    # Try to detect the person within the bed area
                    previous_landmarks = detect_person_in_bed_area(frame, bed_bounding_box, pose)

                    if previous_landmarks is None:
                        print("No pose detected in the bed area during re-initialization.")
                        current_frame += 1
                        continue  # Skip to the next frame after re-detection
                    else:
                        # Continue with the newly detected landmarks
                        current_landmarks = previous_landmarks

                # Update previous_landmarks for the next frame
                previous_landmarks = current_landmarks

                # List of landmarks to monitor
                landmarks_to_monitor = [
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE,
                    mp_pose.PoseLandmark.NOSE
                ]

                for landmark_id in landmarks_to_monitor:
                    # Get the landmark from the pose landmarks
                    landmark = result.pose_landmarks.landmark[landmark_id.value]

                    # Convert normalized coordinates to pixel coordinates in the frame
                    x = landmark.x * width
                    y = landmark.y * height

                    # Calculate the SDF value relative to the bed bounding box
                    x1_bed, y1_bed, x2_bed, y2_bed = bed_bounding_box
                    sdf = point_to_rect_sdf(x, y, x1_bed, y1_bed, x2_bed, y2_bed)
                    sdf_values.append(sdf)

                    # Find the closest point on the bed bounding box
                    closest_x, closest_y = closest_point_on_rect(x, y, x1_bed, y1_bed, x2_bed, y2_bed)

                    # Draw the landmark on the frame
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

                    # Draw a line from the landmark to the closest point on the bed bounding box
                    cv2.line(frame, (int(x), int(y)), (int(closest_x), int(closest_y)), (0, 255, 255), 1)

                    # Display the SDF value next to the landmark
                    cv2.putText(frame, f"SDF: {sdf:.1f}", (int(x) + 10, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Check if all SDFs are positive (person has left the bed)
                if all(sdf > 0 for sdf in sdf_values):
                    current_time = current_frame / frame_rate
                    warning_text = f"Warning: Person has left the bed at {current_time:.2f}s"
                    print(warning_text)
                    cv2.putText(frame, warning_text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # If no pose landmarks are detected, try detecting within the bed area
                print("No pose detected in full frame. Trying detection in bed area.")

                # Attempt to detect the person within the bed area
                previous_landmarks = detect_person_in_bed_area(frame, bed_bounding_box, pose)

                if previous_landmarks is None:
                    print("No pose detected in the bed area during re-initialization.")
                    current_frame += 1
                    continue  # Skip to the next frame after re-detection

            # Optionally, draw the bed bounding box on the frame
            x1_bed, y1_bed, x2_bed, y2_bed = bed_bounding_box
            cv2.rectangle(frame, (x1_bed, y1_bed), (x2_bed, y2_bed), (0, 255, 0), 2)

            # Display the frame with overlays
            cv2.imshow('Video with Pose Detection', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        current_frame += 1

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def detect_person_in_bed_area(frame, bed_bounding_box, pose):
    """
    Detects the person within the bed area using MediaPipe Pose.

    Parameters:
        frame: The current video frame.
        bed_bounding_box: The coordinates of the bed bounding box (x1, y1, x2, y2).
        pose: The initialized MediaPipe Pose object.

    Returns:
        A list of landmarks if detected, otherwise None.
    """
    x1, y1, x2, y2 = bed_bounding_box

    # Crop the frame to the bed bounding box
    cropped_frame = frame[y1:y2, x1:x2]

    # Check if the cropped frame is of sufficient size for MediaPipe
    min_width, min_height = 100, 100  # Minimum dimensions for reliable detection
    if cropped_frame.shape[1] < min_width or cropped_frame.shape[0] < min_height:
        print("Cropped frame is too small for pose detection.")
        return None

    # Convert the cropped frame to RGB as required by MediaPipe
    cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    # Process the cropped frame with MediaPipe Pose
    result = pose.process(cropped_frame_rgb)

    if result.pose_landmarks:
        # Map landmarks back to the original frame
        height, width, _ = cropped_frame.shape
        landmarks = []
        for landmark in result.pose_landmarks.landmark:
            x = landmark.x * width + x1
            y = landmark.y * height + y1
            landmarks.append((x, y))
        return landmarks
    else:
        return None

def main():
    video_path = 'X:/Videos_Hospital/WIN_20240619_16_27_45_Pro.mp4'  # Update this path to your video file
    model_path = 'X:/best.pt'  # Update this path to your YOLOv8 model
    skip_frames = 1  # Adjust as needed for performance

    # Call the function to process the video
    process_video_with_pose_detection_inside_bed(video_path, model_path, skip_frames)
