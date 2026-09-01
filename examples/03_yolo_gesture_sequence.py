import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLOE


# ============================================================
# SETTINGS
# ============================================================

GESTURE_MODEL_PATH = Path(__file__).with_name(
    "gesture_recognizer.task"
)

YOLO_MODEL = "yoloe-26n-seg.pt"


# Change these to the objects you currently have
OBJECT_CLASSES = [
    "cup",
    "cell phone",
    "computer mouse",
    "pin",
]


# ============================================================
# YOLOE
# ============================================================

print("Loading YOLOE...")

yolo_model = YOLOE(YOLO_MODEL)

yolo_model.set_classes(
    OBJECT_CLASSES
)

print("YOLOE ready.")


# ============================================================
# MEDIAPIPE
# ============================================================

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=str(GESTURE_MODEL_PATH)
)

classifier_options = (
    mp.tasks.components.processors.ClassifierOptions(
        score_threshold=0.60,
        category_allowlist=[
            "Open_Palm",
            "Closed_Fist",
        ],
    )
)

gesture_options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    canned_gesture_classifier_options=classifier_options,
)

gesture_recognizer = (
    vision.GestureRecognizer.create_from_options(
        gesture_options
    )
)

print("MediaPipe ready.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


# ============================================================
# PROGRAM STATE
# ============================================================

MODE_OBJECT = "OBJECT"
MODE_GESTURE = "GESTURE"

mode = MODE_OBJECT

confirmed_object = None

start_time = time.perf_counter()
last_timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    current_object = None
    object_confidence = 0.0

    detected_gesture = None
    gesture_confidence = 0.0
    command = None


    # ========================================================
    # MODE 1: OBJECT RECOGNITION
    # ========================================================

    if mode == MODE_OBJECT:

        results = yolo_model.predict(
            frame,
            imgsz=320,
            conf=0.20,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        # YOLO draws its own boxes/masks
        display_frame = result.plot()


        # ----------------------------------------------------
        # Find highest-confidence object
        # ----------------------------------------------------

        if result.boxes is not None and len(result.boxes) > 0:

            best_index = int(
                result.boxes.conf.argmax().item()
            )

            class_id = int(
                result.boxes.cls[best_index].item()
            )

            object_confidence = float(
                result.boxes.conf[best_index].item()
            )

            current_object = result.names[class_id]


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "MODE: OBJECT RECOGNITION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if current_object:

            text = (
                f"Detected: {current_object} "
                f"({object_confidence:.2f})"
            )

            cv2.putText(
                display_frame,
                text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                "Press C to confirm object",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        else:

            cv2.putText(
                display_frame,
                "Show an object to the camera",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


    # ========================================================
    # MODE 2: GESTURE RECOGNITION
    # ========================================================

    elif mode == MODE_GESTURE:

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms


        # ----------------------------------------------------
        # MediaPipe inference
        # ----------------------------------------------------

        result = gesture_recognizer.recognize_for_video(
            mp_image,
            timestamp_ms,
        )


        if result.gestures and result.gestures[0]:

            gesture = result.gestures[0][0]

            detected_gesture = gesture.category_name
            gesture_confidence = gesture.score


            # -----------------------------------------------
            # Gesture → future robot command
            # -----------------------------------------------

            if detected_gesture == "Open_Palm":

                command = "OPEN"


            elif detected_gesture == "Closed_Fist":

                command = "CLOSE"


        # ----------------------------------------------------
        # Draw hand box
        # ----------------------------------------------------

        if result.hand_landmarks:

            height, width, _ = frame.shape

            hand = result.hand_landmarks[0]

            xs = [
                int(point.x * width)
                for point in hand
            ]

            ys = [
                int(point.y * height)
                for point in hand
            ]

            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)

            x2 = min(max(xs) + 20, width)
            y2 = min(max(ys) + 20, height)

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "MODE: GESTURE CONTROL",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


        cv2.putText(
            display_frame,
            f"Object: {confirmed_object}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


        if detected_gesture:

            cv2.putText(
                display_frame,
                (
                    f"Gesture: {detected_gesture} "
                    f"({gesture_confidence:.2f})"
                ),
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


        if command:

            cv2.putText(
                display_frame,
                f"COMMAND: {command}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
            )


        cv2.putText(
            display_frame,
            "Open palm = OPEN | Fist = CLOSE",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            "Press R to recognize another object",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )


    # ========================================================
    # COMMON UI
    # ========================================================

    cv2.putText(
        display_frame,
        "Q = Quit",
        (20, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    cv2.imshow(
        "Sensitive Gripper 2026",
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Confirm detected object
    # --------------------------------------------------------

    if mode == MODE_OBJECT:

        if key == ord("c"):

            if current_object is not None:

                confirmed_object = current_object

                print(
                    f"\nObject confirmed: "
                    f"{confirmed_object}"
                )

                print(
                    "Switching to gesture control..."
                )

                mode = MODE_GESTURE

            else:

                print(
                    "No object detected. "
                    "Nothing to confirm."
                )


    # --------------------------------------------------------
    # Return to object detection
    # --------------------------------------------------------

    elif mode == MODE_GESTURE:

        if key == ord("r"):

            print(
                "\nReturning to object recognition..."
            )

            confirmed_object = None

            mode = MODE_OBJECT


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

gesture_recognizer.close()

cv2.destroyAllWindows()