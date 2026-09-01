from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from ultralytics import YOLOE


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

EMOTION_PROJECT = BASE_DIR / "emotion_model"

sys.path.insert(0, str(EMOTION_PROJECT))

from app.inference import load_emotion_model, predict_emotion
from src.config import EMOTION_LABELS


CHECKPOINT = (
    EMOTION_PROJECT
    / "outputs"
    / "checkpoints"
    / "best_efficientnet_b2.pt"
)

CASCADE_PATH = (
    BASE_DIR
    / "haarcascade_frontalface_default.xml"
)


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

YOLO_MODEL = "yoloe-26n-seg.pt"

OBJECT_CLASSES = [
    "cup",
    "cell phone",
    "computer mouse",
    "pin",
]

YOLO_CONFIDENCE = 0.20

# Time between confirming the object and grasp command
PRE_GRASP_DELAY = 2.0

# Time during which we observe human feedback
FEEDBACK_TIME = 5.0

# Emotion smoothing
SMOOTHING = 5

# Happiness must be reasonably confident
HAPPINESS_THRESHOLD = 0.60

# Require several consecutive happy predictions
HAPPINESS_CONFIRM_FRAMES = 3

IMAGE_SIZE = 224


# ============================================================
# PROGRAM STATES
# ============================================================

STATE_OBJECT = "OBJECT"
STATE_PREPARE = "PREPARE"
STATE_FEEDBACK = "FEEDBACK"
STATE_RESULT = "RESULT"

state = STATE_OBJECT


# ============================================================
# LOAD YOLOE
# ============================================================

print("Loading YOLOE...")

yolo_model = YOLOE(YOLO_MODEL)

yolo_model.set_classes(
    OBJECT_CLASSES
)

print("YOLOE ready.")


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

print("Loading emotion recognition model...")

emotion_model, device = load_emotion_model(
    checkpoint_path=CHECKPOINT,
    model_name="efficientnet_b2",
    device="auto",
)

print("Emotion model ready.")
print(f"Emotion model device: {device}")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

if not CASCADE_PATH.exists():
    raise FileNotFoundError(
        f"Haar cascade not found:\n{CASCADE_PATH}"
    )

face_detector = cv2.CascadeClassifier(
    str(CASCADE_PATH)
)

if face_detector.empty():
    raise RuntimeError(
        "Could not load OpenCV Haar face detector."
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_INDEX}"
    )


# ============================================================
# STATE VARIABLES
# ============================================================

confirmed_object = None

state_start_time = time.perf_counter()

probability_buffer = deque(
    maxlen=SMOOTHING
)

happy_counter = 0
happy_detected = False

final_result = None


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_experiment():

    global state
    global confirmed_object
    global state_start_time

    global happy_counter
    global happy_detected
    global final_result

    state = STATE_OBJECT

    confirmed_object = None

    happy_counter = 0
    happy_detected = False

    final_result = None

    probability_buffer.clear()

    state_start_time = time.perf_counter()

    print("\n--------------------------------")
    print("Experiment reset")
    print("Waiting for new object...")
    print("--------------------------------")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    current_time = time.perf_counter()

    current_object = None
    object_confidence = 0.0

    emotion_label = "Waiting"
    emotion_confidence = 0.0


    # ========================================================
    # STATE 1 — OBJECT RECOGNITION
    # ========================================================

    if state == STATE_OBJECT:

        results = yolo_model.predict(
            frame,
            imgsz=320,
            conf=YOLO_CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        display_frame = result.plot()


        # ----------------------------------------------------
        # Select highest-confidence detected object
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            best_index = int(
                result.boxes.conf.argmax().item()
            )

            class_id = int(
                result.boxes.cls[
                    best_index
                ].item()
            )

            object_confidence = float(
                result.boxes.conf[
                    best_index
                ].item()
            )

            current_object = (
                result.names[class_id]
            )


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "MODE: OBJECT RECOGNITION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )


        if current_object:

            cv2.putText(
                display_frame,
                (
                    f"Detected: {current_object} "
                    f"({object_confidence:.2f})"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                display_frame,
                "Press C to confirm",
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
    # STATE 2 — PREPARE FOR GRASP
    # ========================================================

    elif state == STATE_PREPARE:

        elapsed = (
            current_time - state_start_time
        )

        remaining = max(
            PRE_GRASP_DELAY - elapsed,
            0
        )


        cv2.putText(
            display_frame,
            "OBJECT CONFIRMED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Object: {confirmed_object}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            display_frame,
            "Preparing grasp...",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Closing in: {remaining:.1f} s",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )


        # ----------------------------------------------------
        # After delay → issue CLOSE
        # ----------------------------------------------------

        if elapsed >= PRE_GRASP_DELAY:

            print("\n>>> COMMAND: CLOSE")

            # FUTURE:
            #
            # serial_port.write(b"CLOSE\n")
            #
            # For now we only print/display the command.

            state = STATE_FEEDBACK

            state_start_time = time.perf_counter()

            probability_buffer.clear()

            happy_counter = 0
            happy_detected = False

            print(
                f"Looking for human feedback "
                f"for {FEEDBACK_TIME:.0f} seconds..."
            )


    # ========================================================
    # STATE 3 — HUMAN EMOTION FEEDBACK
    # ========================================================

    elif state == STATE_FEEDBACK:

        elapsed = (
            current_time - state_start_time
        )

        remaining = max(
            FEEDBACK_TIME - elapsed,
            0
        )


        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(48, 48),
        )


        # ----------------------------------------------------
        # PROCESS LARGEST FACE
        # ----------------------------------------------------

        if len(faces) > 0:

            x, y, w, h = max(
                faces,
                key=lambda box:
                    box[2] * box[3]
            )


            margin = int(
                0.15 * max(w, h)
            )


            x1 = max(
                x - margin,
                0
            )

            y1 = max(
                y - margin,
                0
            )

            x2 = min(
                x + w + margin,
                frame.shape[1]
            )

            y2 = min(
                y + h + margin,
                frame.shape[0]
            )


            # ------------------------------------------------
            # CROP FACE
            # ------------------------------------------------

            face_bgr = frame[
                y1:y2,
                x1:x2
            ]

            face_rgb = cv2.cvtColor(
                face_bgr,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # EMOTION INFERENCE
            # ------------------------------------------------

            prediction = predict_emotion(
                model=emotion_model,
                device=device,
                image=face_rgb,
                image_size=IMAGE_SIZE,
                imagenet_norm=True,
            )


            probability_buffer.append(
                prediction[
                    "probability_array"
                ]
            )


            # ------------------------------------------------
            # SMOOTH PREDICTIONS
            # ------------------------------------------------

            smoothed = np.mean(
                np.stack(
                    probability_buffer,
                    axis=0
                ),
                axis=0,
            )


            label_id = int(
                smoothed.argmax()
            )

            emotion_label = (
                EMOTION_LABELS[
                    label_id
                ]
            )

            emotion_confidence = float(
                smoothed[
                    label_id
                ]
            )


            # ------------------------------------------------
            # CHECK FOR HAPPINESS
            # ------------------------------------------------

            if (
                emotion_label == "happiness"
                and
                emotion_confidence
                >= HAPPINESS_THRESHOLD
            ):

                happy_counter += 1

            else:

                happy_counter = 0


            if (
                happy_counter
                >= HAPPINESS_CONFIRM_FRAMES
            ):

                happy_detected = True


            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                2,
            )


        else:

            probability_buffer.clear()
            happy_counter = 0

            emotion_label = "No face"
            emotion_confidence = 0.0


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "COMMAND: CLOSE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
        )


        cv2.putText(
            display_frame,
            "WAITING FOR HUMAN FEEDBACK",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            f"Object: {confirmed_object}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            (
                f"Emotion: {emotion_label} "
                f"({emotion_confidence:.2f})"
            ),
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            f"Feedback time: {remaining:.1f} s",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


        if happy_detected:

            cv2.putText(
                display_frame,
                "HAPPINESS DETECTED",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )


        # ----------------------------------------------------
        # END FEEDBACK WINDOW
        # ----------------------------------------------------

        if elapsed >= FEEDBACK_TIME:

            if happy_detected:

                final_result = "SUCCESS"

                print(
                    ">>> HUMAN FEEDBACK: SUCCESS"
                )

            else:

                final_result = "FAILED"

                print(
                    ">>> HUMAN FEEDBACK: FAILED"
                )


            state = STATE_RESULT

            state_start_time = (
                time.perf_counter()
            )


    # ========================================================
    # STATE 4 — RESULT
    # ========================================================

    elif state == STATE_RESULT:

        cv2.putText(
            display_frame,
            "GRASP EVALUATION",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            f"Object: {confirmed_object}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )


        cv2.putText(
            display_frame,
            f"RESULT: {final_result}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
        )


        if final_result == "SUCCESS":

            cv2.putText(
                display_frame,
                "Positive human feedback detected",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        else:

            cv2.putText(
                display_frame,
                "No positive feedback detected",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


        cv2.putText(
            display_frame,
            "Press R to try another object",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


    # ========================================================
    # COMMON UI
    # ========================================================

    cv2.putText(
        display_frame,
        "R = Reset   Q = Quit",
        (20, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )


    cv2.imshow(
        "05 - YOLO + Emotion Feedback",
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # C = CONFIRM OBJECT
    # --------------------------------------------------------

    if (
        state == STATE_OBJECT
        and
        key == ord("c")
    ):

        if current_object is not None:

            confirmed_object = current_object

            print(
                f"\nObject confirmed: "
                f"{confirmed_object}"
            )

            state = STATE_PREPARE

            state_start_time = (
                time.perf_counter()
            )

        else:

            print(
                "No object detected. "
                "Nothing to confirm."
            )


    # --------------------------------------------------------
    # R = RESET
    # --------------------------------------------------------

    if key == ord("r"):

        reset_experiment()


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()