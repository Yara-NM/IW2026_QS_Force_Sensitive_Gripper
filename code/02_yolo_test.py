from ultralytics import YOLOE
import cv2


# ---------------------------------------------------------
# Load the smallest current YOLOE model
# ---------------------------------------------------------

model = YOLOE("yoloe-26n-seg.pt")


# ---------------------------------------------------------
# Tell the model what we want to detect
# No training required.
# ---------------------------------------------------------

model.set_classes([
    "cup",
    "phone",
    "pen",
])


# ---------------------------------------------------------
# Open webcam
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # -----------------------------------------------------
    # YOLOE inference
    # -----------------------------------------------------

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.20,
        device="cpu",
        verbose=False,
    )


    # -----------------------------------------------------
    # Draw detections
    # -----------------------------------------------------

    annotated_frame = results[0].plot()


    cv2.putText(
        annotated_frame,
        "YOLOE-26n | Q = quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


    cv2.imshow(
        "YOLOE Object Recognition",
        annotated_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()