import cv2
import numpy as np
from ultralytics import YOLO

def process_video_with_static_bounding_box(video_path, model_path, skip_frames=None, new_width=None, new_height=None, threshold=1):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        exit()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    # For warning detection
    max_avg_freq = None  # Initialize the maximum average frequency
    avg_frequencies = []
    tempos = []

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

        # Ensure coordinates are within image boundaries (in case of any changes)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Crop the frame to the bounding box
        cropped_frame = frame[y1:y2, x1:x2]

        # Convert to grayscale
        gray_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Resize if specified
        if new_width and new_height:
            gray_cropped = cv2.resize(gray_cropped, (new_width, new_height))

        # Calculate FFT
        f = np.fft.fft2(gray_cropped)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Add 1 to avoid log(0)

        # Compute average frequency magnitude for warning detection
        avg_freq = np.mean(magnitude_spectrum)
        avg_frequencies.append(avg_freq)

        # Update the maximum average frequency
        if max_avg_freq is None or avg_freq > max_avg_freq:
            max_avg_freq = avg_freq

        # Store the time
        tempo_atual = current_frame / frame_rate
        tempos.append(tempo_atual)

        # Check for significant drop from the maximum average frequency magnitude
        drop = max_avg_freq - avg_freq
        if drop > threshold:
            warning_text = f"Warning: Significant drop at {tempo_atual:.2f}s"
            print(warning_text)
            # Overlay warning on the frame
            cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Optionally, draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'X:/Videos_Hospital/WIN_20240619_16_27_45_Pro.mp4'  # Update this path
    model_path = 'X:/best.pt'  # Update this path
    skip_frames = 4  # Skip frames to speed up processing
    new_width = 320  # Resize width
    new_height = 240  # Resize height
    threshold = 0.35  # Threshold for warning

    process_video_with_static_bounding_box(
        video_path, model_path, skip_frames, new_width, new_height, threshold)
