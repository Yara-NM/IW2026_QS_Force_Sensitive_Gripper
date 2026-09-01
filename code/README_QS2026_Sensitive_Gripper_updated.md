# QS 2026 — Vision-Guided Force-Sensitive Robotic Gripper

This folder contains the computer-vision prototypes for the **Skoltech Innovation Workshop 2026 — Sensitive Gripper** activity.

The current concept uses one webcam in two sequential modes:

1. **Object recognition with YOLOE** — the robot identifies what object is in front of it.
2. **Human confirmation** — the user confirms the detected object from the keyboard.
3. **Gesture recognition with MediaPipe** — the same webcam then observes the operator's hand.
4. **Future integration** — `OPEN` / `CLOSE` commands will be sent to the ESP32, while the force sensor provides the low-level safety limit.

The main educational idea is:

> **Vision understands the object, the human provides the intention, and force sensing keeps the grasp safe.**

---

## 1. Software Overview

### OpenCV

**OpenCV** is the computer-vision library used to access the laptop webcam, read frames, draw information on the video, and display the result.

Official website:

https://opencv.org/

Python package:

```bash
pip install opencv-python
```

Typical use in our code:

```python
import cv2

cap = cv2.VideoCapture(0)
success, frame = cap.read()
cv2.imshow("Camera", frame)
```

OpenCV does **not** recognize our objects or gestures by itself in this project. It mainly provides the camera/video interface used by both MediaPipe and YOLOE.

---

### MediaPipe Gesture Recognizer

**MediaPipe** is Google's framework for real-time perception tasks.

For this workshop we use the current **MediaPipe Tasks Gesture Recognizer**, rather than the older `mp.solutions.hands` workflow used in the previous workshop.

Official Gesture Recognizer documentation:

https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python

Official MediaPipe Python setup:

https://developers.google.com/edge/mediapipe/solutions/setup_python

Python package:

```bash
pip install mediapipe
```

The pretrained Gesture Recognizer can recognize several built-in gestures, including:

- `Open_Palm`
- `Closed_Fist`
- `Pointing_Up`
- `Thumb_Up`
- `Thumb_Down`
- `Victory`
- `ILoveYou`

For our gripper prototype we currently use only:

```text
Open_Palm   -> OPEN
Closed_Fist -> CLOSE
```

The recognizer also provides hand landmarks, but our current control logic does not need to manually calculate the distance between individual fingers.

The model file used by the code is:

```text
gesture_recognizer.task
```

Official model download:

https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

Place this file in the same folder as the Python scripts.

---

### Ultralytics YOLOE

**YOLO** stands for **You Only Look Once**. It is a family of neural-network models designed for real-time computer vision.

For this workshop we are testing **YOLOE**, an open-vocabulary version of YOLO. Instead of being restricted to a fixed list of classes, YOLOE can be given text prompts describing the objects we want to recognize.

Official Ultralytics website:

https://www.ultralytics.com/

Official Ultralytics documentation:

https://docs.ultralytics.com/

Official YOLOE documentation:

https://docs.ultralytics.com/models/yoloe/

Official installation guide:

https://docs.ultralytics.com/quickstart/

Install or update Ultralytics:

```bash
pip install -U ultralytics
```

We currently use the lightweight YOLOE nano model:

```text
yoloe-26n-seg.pt
```

Example:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")

model.set_classes([
    "cup",
    "cell phone",
    "computer mouse",
    "pin",
])
```

The important concept is that these are **text prompts**. We can change them without training a new model.

For example:

```python
model.set_classes([
    "egg",
    "stone",
    "sponge",
])
```

can later be used for the workshop objects.

### First YOLOE run

The first run may take longer because Ultralytics may download:

- the YOLOE model weights;
- the text-prompt encoder required by `set_classes()`;
- additional CLIP-related dependencies.

For YOLOE-26 text prompting, the text encoder is currently approximately 254 MB. Internet access is therefore useful during the **initial setup**.

For the actual workshop, all required models should be downloaded and tested on the laptops **before the participants arrive**.

---

## 2. Recommended Python Environment

The prototype has been tested with a Python virtual environment.

Current development version:

```text
Python 3.12
```

### Windows

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python
python -m pip install -U ultralytics
```

Once the environment is activated, normal commands can be used:

```powershell
python 01_gesture_test.py
```

There is no need to repeatedly write the complete `.venv\Scripts\python.exe` path.

---

### Linux / Ubuntu

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the libraries:

```bash
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python
python -m pip install -U ultralytics
```

Run a script:

```bash
python 01_gesture_test.py
```

The MediaPipe/OpenCV/Ultralytics approach is intended to work on both **Windows and Linux**.

The main operating-system difference relevant to our scripts is usually the camera backend. The portable option is:

```python
cv2.VideoCapture(0)
```

On Windows, DirectShow can also be explicitly selected:

```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

---

## 3. Current Project Files

Current project structure:

```text
code/
│
├── .venv/
├── emotion_model/
│   ├── app/
│   ├── src/
│   └── outputs/
│       └── checkpoints/
│           └── best_efficientnet_b2.pt
│
├── 00_camera_test.py
├── 01_gesture_test.py
├── 02_yolo_test.py
├── 03_yolo_gesture_sequence.py
├── 04_emotion_test.py
├── 05_yolo_emotion_sequence.py
│
├── gesture_recognizer.task
├── haarcascade_frontalface_default.xml
├── mobileclip2_b.ts
└── yoloe-26n-seg.pt
```

### Important local model/data files

- `gesture_recognizer.task` — pretrained MediaPipe Gesture Recognizer model.
- `yoloe-26n-seg.pt` — lightweight YOLOE nano model used for object recognition.
- `mobileclip2_b.ts` — text encoder used by YOLOE when working with text prompts.
- `haarcascade_frontalface_default.xml` — OpenCV Haar cascade used to locate faces before facial-expression classification.
- `emotion_model/outputs/checkpoints/best_efficientnet_b2.pt` — pretrained EfficientNet-B2 facial-expression classifier from the teammate project.

Keeping these files locally is useful for the workshop because it reduces dependence on internet access and package/model downloads during the session.

---

## 4. What Each Script Does

### `00_camera_test.py`

**Purpose:** verify that OpenCV can access the webcam before testing any AI model.

Pipeline:

```text
Webcam
  |
OpenCV
  |
Display video
```

This script helps separate camera problems from AI/model problems.

Run:

```bash
python 00_camera_test.py
```

Typical controls:

```text
Q -> quit
```

If the webcam cannot be opened, check:

- whether another application is using the camera;
- camera permissions;
- camera index `0`, `1`, or `2`;
- Windows/Linux camera backend settings.

---

### `01_gesture_test.py`

**Purpose:** test the Google MediaPipe Gesture Recognizer independently.

Pipeline:

```text
Webcam
   |
OpenCV frame
   |
MediaPipe Gesture Recognizer
   |
Open_Palm / Closed_Fist
   |
OPEN / CLOSE text command
```

Current mapping:

```text
Open_Palm   -> OPEN
Closed_Fist -> CLOSE
```

At this stage the command is displayed only on the laptop. It is **not yet sent to the physical gripper**.

This is intentional: every perception component is tested independently before hardware integration.

Run:

```bash
python 01_gesture_test.py
```

---

### `02_yolo_test.py`

**Purpose:** test YOLOE open-vocabulary object recognition independently.

Pipeline:

```text
Webcam
   |
OpenCV frame
   |
YOLOE
   |
Object class + confidence
```

Current test classes:

```python
OBJECT_CLASSES = [
    "cup",
    "cell phone",
    "computer mouse",
    "pin",
]
```

These are temporary development objects and can be changed freely.

Example future workshop classes might include:

```python
OBJECT_CLASSES = [
    "egg",
    "stone",
    "sponge",
]
```

The purpose of this script is to determine:

- whether YOLOE runs smoothly on the laptop;
- which objects are recognized reliably;
- which text prompts give the best results;
- whether the selected model is light enough for student laptops.

Run:

```bash
python 02_yolo_test.py
```

---

### `03_yolo_gesture_sequence.py`

**Purpose:** combine YOLOE and MediaPipe while using only **one webcam**.

The two models are deliberately used **sequentially**, rather than processing every frame simultaneously.

Overall workflow:

```text
START
  |
  v
OBJECT RECOGNITION MODE
  |
  |  YOLOE detects the object
  v
Detected object + confidence
  |
  |  User checks result
  |  Press C
  v
OBJECT CONFIRMED
  |
  v
GESTURE CONTROL MODE
  |
  |  MediaPipe observes the hand
  |
  +---- Open_Palm ----> OPEN
  |
  +---- Closed_Fist --> CLOSE
  |
  |  Press R
  v
Return to object recognition
```

Keyboard controls:

```text
C -> confirm the current YOLOE object
R -> return to object recognition mode
Q -> quit
```

### Why use sequential modes?

There is only one camera, and we do not actually need YOLOE and MediaPipe to process the image at the same time.

Sequential processing has several advantages:

1. Lower CPU load.
2. Better compatibility with ordinary student laptops.
3. Easier debugging.
4. The operator does not need to show their hand while the object is being analyzed.
5. YOLOE does not continuously change the object decision after it has been confirmed.

The interaction becomes:

```text
Robot: "What object is this?"
        |
       YOLOE
        |
User: "Yes, that is correct."
        |
   keyboard C
        |
Robot: "What do you want me to do?"
        |
    MediaPipe
        |
     hand gesture
```

This is also an example of **human supervisory control**: the AI proposes an interpretation, but the human confirms it before the robot acts.

---

### `04_emotion_test.py`

**Purpose:** test the teammate's pretrained facial-expression neural network independently before combining it with YOLOE.

The model comes from:

https://github.com/CaptainEv1dence/dl-project

The current pipeline is:

```text
Webcam
   |
OpenCV
   |
Haar face detector
   |
Crop largest detected face
   |
EfficientNet-B2 neural network
   |
Facial-expression probabilities
   |
Smoothed prediction
```

The neural network predicts seven labels:

```text
anger
disgust
fear
happiness
sadness
surprise
neutral
```

The workshop code does **not train the network**. It loads the pretrained checkpoint:

```text
emotion_model/outputs/checkpoints/best_efficientnet_b2.pt
```

The Haar cascade file:

```text
haarcascade_frontalface_default.xml
```

is kept directly in the project folder rather than relying on the OpenCV package to contain it. This makes the setup more portable between laptops.

Run:

```bash
python 04_emotion_test.py
```

### Current practical observation

During quick local testing, facial-expression prediction is noticeably less reliable than MediaPipe gesture recognition. **Happiness is currently the most consistently recognized expression** in our setup.

For this reason, the current experiment treats the model as a **facial-expression feedback classifier**, not as a perfect measurement of a person's true emotional state.

```text
Camera sees a facial expression
        |
Neural network predicts a label
        |
Application interprets the label as feedback
```

The robot is not literally "reading emotions"; it is using a learned visual classifier as an HRI feedback channel.

---

### `05_yolo_emotion_sequence.py`

**Purpose:** combine YOLOE object recognition with neural-network facial-expression feedback while still using only one webcam.

This is the second experimental branch and does **not use MediaPipe**.

The current workflow is:

```text
START
  |
  v
OBJECT RECOGNITION MODE
  |
  |  YOLOE detects object
  v
Detected object + confidence
  |
  |  User verifies prediction
  |  Press C
  v
OBJECT CONFIRMED
  |
  |  Wait ~2 seconds
  v
COMMAND: CLOSE
  |
  |  Currently displayed only
  |  Later sent to ESP32
  v
5 SECOND FEEDBACK WINDOW
  |
  |  Face detector + EfficientNet-B2
  v
Is happiness detected reliably?
  |
  +---- YES ----> SUCCESS
  |
  +---- NO -----> FAILED
```

Keyboard controls:

```text
C -> confirm the currently detected YOLOE object
R -> reset the complete experiment and return to object recognition
Q -> quit
```

The current prototype uses:

```text
PRE_GRASP_DELAY = 2 seconds
FEEDBACK_TIME   = 5 seconds
```

After the object is confirmed, the software waits briefly and then displays:

```text
COMMAND: CLOSE
```

No physical motor command is sent yet.

The camera then switches from object recognition to facial-expression feedback. If sufficiently stable `happiness` is detected during the feedback window, the attempt is stored as:

```text
SUCCESS
```

Otherwise:

```text
FAILED
```

### Why require multiple happy predictions?

Facial-expression classification can fluctuate from frame to frame. The code therefore uses:

- temporal probability smoothing;
- a confidence threshold;
- several consecutive positive predictions.

This prevents a single accidental `happiness` frame from immediately declaring the grasp successful.

### Educational interpretation

This branch explores a different human-robot interaction idea from `03`:

```text
03: YOLOE + MediaPipe
    Object understanding + human COMMAND

05: YOLOE + neural network
    Object understanding + human FEEDBACK
```

In `03`, the human tells the robot **what action to perform**.

In `05`, the robot performs the planned action and then uses the human's facial-expression signal to evaluate **whether the action was accepted as successful**.

This gives the workshop team two alternative interaction stories that can be compared before choosing the final experiment.

---

## 5. Planned Hardware Integration

The computer-vision code is currently intentionally separated from the hardware.

The next architecture is expected to support two alternative HRI branches.

### Branch A — Gesture-controlled grasping

```text
YOLOE object recognition
        |
Human confirms object
        |
MediaPipe gesture
        |
OPEN / CLOSE
        |
USB/Serial
        |
ESP32
   +----+----+
   |         |
  FSR    Dynamixel
```

Possible logic:

```text
1. Recognize object.
2. Human confirms object.
3. Select an appropriate force profile.
4. Wait for a hand gesture.
5. OPEN or CLOSE the gripper.
6. During closing, the ESP32 monitors the force sensor.
7. Stop the motor when the force limit is reached.
```

### Branch B — Affective-feedback grasping

```text
YOLOE object recognition
        |
Human confirms object
        |
CLOSE command
        |
USB/Serial
        |
ESP32 + FSR + Dynamixel
        |
Grasp attempt
        |
Facial-expression NN
        |
SUCCESS / FAILED feedback
```

A future extension could use negative feedback to release, modify the force limit, or request another grasp attempt.

A key design principle for both branches is that **force-safety control should remain on the ESP32**, rather than depending on continuous laptop communication.

---

## 6. Future Workshop Object Logic

The final workshop objects are still being evaluated.

One possible concept is:

```text
Object type       Example       Grasp strategy
------------------------------------------------
Fragile           Egg           Low force limit
Deformable        Sponge        Medium/adaptive limit
Rigid             Stone         Higher force limit
```

YOLOE provides **visual context**, while the force-sensitive resistor provides **physical feedback**.

The robot therefore combines:

```text
VISION
"What am I handling?"

      +

HUMAN INTENT
"What should I do?"

      +

TOUCH
"How hard am I squeezing?"
```

---

## 7. Important Workshop Preparation Notes for TAs

To keep the activity achievable in approximately four hours of student development time:

- Test every laptop before the workshop.
- Create the Python virtual environment in advance.
- Install MediaPipe, OpenCV, Ultralytics, PyTorch, and torchvision in advance.
- Download `gesture_recognizer.task` in advance.
- Run YOLOE at least once in advance so required model files and text encoders are already available.
- Keep `yoloe-26n-seg.pt` and `mobileclip2_b.ts` available locally.
- Keep `haarcascade_frontalface_default.xml` in the project folder.
- Verify that `emotion_model/outputs/checkpoints/best_efficientnet_b2.pt` is present.
- Run `04_emotion_test.py` once on each laptop if the facial-feedback branch may be used.
- Test the webcam.
- Test USB/serial communication with the ESP32.
- Avoid spending student workshop time debugging package installation.
- Keep the vision components modular so they can be tested independently.
- Keep a known-working copy of every script for recovery if a team gets stuck.

The workshop should focus on **robotics integration and experimentation**, not package installation.

---

## 8. Quick Installation Checklist

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python
python -m pip install -U ultralytics
```

Ultralytics normally installs the required PyTorch components. Verify the full environment with:

```bash
python -c "import cv2, mediapipe, ultralytics, torch, torchvision, numpy, PIL; print('Environment OK'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected minimum result:

```text
Environment OK
```

The emotion model can run on CPU. CUDA is optional and should not be required for the workshop unless every target laptop has been prepared and tested with it.

---

## 9. Useful Links

### MediaPipe / Google

Gesture Recognizer for Python:

https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python

MediaPipe Python setup:

https://developers.google.com/edge/mediapipe/solutions/setup_python

Gesture Recognizer model:

https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

MediaPipe GitHub:

https://github.com/google-ai-edge/mediapipe

### YOLO / Ultralytics

Ultralytics:

https://www.ultralytics.com/

Documentation:

https://docs.ultralytics.com/

Installation / Quickstart:

https://docs.ultralytics.com/quickstart/

YOLOE documentation:

https://docs.ultralytics.com/models/yoloe/

Ultralytics GitHub:

https://github.com/ultralytics/ultralytics

### OpenCV

OpenCV:

https://opencv.org/

OpenCV Python tutorials:

https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

OpenCV Haar cascades:

https://github.com/opencv/opencv/tree/master/data/haarcascades

### Facial-Expression Neural Network

Teammate project:

https://github.com/CaptainEv1dence/dl-project

PyTorch:

https://pytorch.org/

Torchvision:

https://pytorch.org/vision/stable/index.html

---

## 10. Current Development Status

```text
[✓] Webcam access with OpenCV
[✓] MediaPipe Gesture Recognizer
[✓] Open_Palm -> OPEN
[✓] Closed_Fist -> CLOSE
[✓] YOLOE open-vocabulary object recognition
[✓] Sequential YOLOE -> confirmation -> MediaPipe workflow
[✓] Pretrained EfficientNet-B2 facial-expression model loads and runs
[✓] Local Haar face detector integrated
[✓] Facial-expression webcam test
[✓] Sequential YOLOE -> confirmation -> CLOSE -> emotion-feedback workflow
[ ] Select final workshop objects
[ ] Decide final branch: gesture command vs facial-expression feedback
[ ] Map object class to force profile
[ ] Python -> ESP32 serial communication
[ ] Dynamixel integration
[ ] FSR force-limit integration
[ ] Full physical experiment
```

---

**Project:** QS 2026 Sensitive Gripper  
**Workshop:** Skoltech Innovation Workshop 2026
