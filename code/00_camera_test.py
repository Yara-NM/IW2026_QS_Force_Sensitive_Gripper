import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Mirror the image so interaction feels natural
    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
