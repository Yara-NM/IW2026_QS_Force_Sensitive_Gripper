import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find model:\n{MODEL_PATH}\n"
        "Place gesture_recognizer.task in the same folder as this script."
    )


# ----------------------------------------------------
# Create MediaPipe Gesture Recognizer
# ----------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

classifier_options = mp.tasks.components.processors.ClassifierOptions(
    score_threshold=0.55,
    category_allowlist=[
        "Open_Palm",
        "Closed_Fist"
    ]
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    canned_gesture_classifier_options=classifier_options
)


# ----------------------------------------------------
# Open webcam
# ----------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. "
        "Try changing camera index 0 to 1."
    )


# Used because MediaPipe VIDEO mode requires
# increasing timestamps.
start_time = time.perf_counter()
last_timestamp_ms = 0


# ----------------------------------------------------
# Main loop
# ----------------------------------------------------

with vision.GestureRecognizer.create_from_options(options) as recognizer:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Create monotonically increasing timestamp
        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        # Run gesture recognition
        result = recognizer.recognize_for_video(
            mp_image,
            timestamp_ms
        )


        # ------------------------------------------------
        # Default state
        # ------------------------------------------------

        gesture_name = "No gesture"
        confidence = 0.0
        command = "WAIT"


        # ------------------------------------------------
        # Read recognized gesture
        # ------------------------------------------------

        if result.gestures and result.gestures[0]:

            gesture = result.gestures[0][0]

            gesture_name = gesture.category_name
            confidence = gesture.score

            if gesture_name == "Open_Palm":
                command = "OPEN"

            elif gesture_name == "Closed_Fist":
                command = "CLOSE"


        # ------------------------------------------------
        # Draw bounding box around detected hand
        # ------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            xs = [
                int(landmark.x * width)
                for landmark in hand
            ]

            ys = [
                int(landmark.y * height)
                for landmark in hand
            ]

            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)

            x2 = min(max(xs) + 20, width)
            y2 = min(max(ys) + 20, height)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ------------------------------------------------
        # Display results
        # ------------------------------------------------

        text1 = (
            f"Gesture: {gesture_name} "
            f"({confidence:.2f})"
        )

        text2 = f"Command: {command}"

        cv2.putText(
            frame,
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            text2,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Gripper Gesture Control",
            frame
        )


        # ------------------------------------------------
        # Exit
        # ------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()