from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================
# PATH TO TEAMMATE'S EMOTION PROJECT
# ============================================================

EMOTION_PROJECT = Path(__file__).parent / "emotion_model"

if not EMOTION_PROJECT.exists():
    raise FileNotFoundError(
        f"Emotion project not found:\n{EMOTION_PROJECT}"
    )

# Makes app/ and src/ from the teammate's project importable
sys.path.insert(0, str(EMOTION_PROJECT))


# ============================================================
# IMPORT THEIR EXISTING MODEL CODE
# ============================================================

from app.inference import load_emotion_model, predict_emotion
from src.config import EMOTION_LABELS


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT = (
    EMOTION_PROJECT
    / "outputs"
    / "checkpoints"
    / "best_efficientnet_b2.pt"
)

CAMERA_INDEX = 0

IMAGE_SIZE = 224

# Average the last few predictions.
# This reduces emotion flickering.
SMOOTHING = 5


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


# ============================================================
# LOAD NEURAL NETWORK
# ============================================================

print("Loading emotion recognition model...")

model, device = load_emotion_model(
    checkpoint_path=CHECKPOINT,
    model_name="efficientnet_b2",
    device="auto",
)

print("Emotion model loaded.")
print(f"Device: {device}")


# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = Path(__file__).with_name(
    "haarcascade_frontalface_default.xml"
)

if not cascade_path.exists():
    raise FileNotFoundError(
        f"Haar cascade not found:\n{cascade_path}"
    )

face_detector = cv2.CascadeClassifier(
    str(cascade_path)
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
# SMOOTHING BUFFER
# ============================================================

probability_buffer = deque(
    maxlen=SMOOTHING
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # Mirror image for natural interaction
    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

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


    emotion_label = "No face"
    confidence = 0.0


    # --------------------------------------------------------
    # USE THE LARGEST FACE
    # --------------------------------------------------------

    if len(faces) > 0:

        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )


        # Slight margin around face
        margin = int(
            0.15 * max(w, h)
        )

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)

        x2 = min(
            x + w + margin,
            frame.shape[1]
        )

        y2 = min(
            y + h + margin,
            frame.shape[0]
        )


        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face_bgr = frame[
            y1:y2,
            x1:x2
        ]

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # NEURAL NETWORK INFERENCE
        # ----------------------------------------------------

        result = predict_emotion(
            model=model,
            device=device,
            image=face_rgb,
            image_size=IMAGE_SIZE,
            imagenet_norm=True,
        )


        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------

        probability_buffer.append(
            result["probability_array"]
        )

        smoothed_probabilities = np.mean(
            np.stack(
                probability_buffer,
                axis=0
            ),
            axis=0,
        )


        label_id = int(
            smoothed_probabilities.argmax()
        )

        emotion_label = (
            EMOTION_LABELS[label_id]
        )

        confidence = float(
            smoothed_probabilities[label_id]
        )


        # ----------------------------------------------------
        # DRAW FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            2,
        )


    else:

        # Important:
        # Don't keep old emotion predictions
        # after the face disappears.
        probability_buffer.clear()


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        "EMOTION RECOGNITION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )


    cv2.putText(
        frame,
        f"Emotion: {emotion_label}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )


    cv2.putText(
        frame,
        f"Device: {device}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


    cv2.putText(
        frame,
        "Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )


    cv2.imshow(
        "04 - Emotion Recognition",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()