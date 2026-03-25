# 👁️ COMPUTER_VISION-

Welcome to the **COMPUTER_VISION-** repository. This project is a collection of real-time machine learning and computer vision applications built with Python, OpenCV, and MediaPipe, focused on spatial computing, human–computer interaction, and facial landmarking to create touchless interfaces.[page:1]

---

## 🚀 Core Modules

### 1. Gesture-Based Volume Controller (`2.volume_controller.py`)

A real-time audio controller that maps hand movements to system volume.[page:1]

- Uses MediaPipe hand tracking to detect thumb and index finger landmarks.[page:1]  
- Computes Euclidean distance between these points and maps it to audio levels using Pycaw.[page:1]  

**Use case:** Touchless media and system volume control.[page:1]

---

### 2. Interactive 3D Object Viewer  
`5.cube.py`, `6.cube_proto.py`, `7.block.click.py`

Inspired by “Iron Man” HUD concepts, this module renders a dynamic 3D wireframe cube on the camera feed.[page:1]

- Uses rotational logic and matrix transformations to project 3D objects onto a 2D frame.[page:1]  
- Supports two‑handed scaling and rotation via gesture tracking.[page:1]  

**Use case:** Prototyping spatial computing UI and AR-style object manipulation.[page:1]

---

### 3. Blink Counters & Face Detection  
`8.eyedetection.py`, `9.blink_count.py`, `10.blinkcount_mod.py`

A facial mapping system that tracks micro‑expressions and monitors eye states in real time.[page:1]

- Loads `face_landmarker.task` to create a dense facial mesh.[page:1]  
- Computes Eye Aspect Ratio (EAR) from eyelid distances to detect blinks.[page:1]  

**Use case:** Driver drowsiness detection, accessibility tools, blink‑triggered events.[page:1]

---

### 4. Virtual Dials & Hybrid Controls  
`3.dialtest.3.py`, `11.dial.blink.py`

Experimental scripts that fuse multiple detection streams (hand gestures + blink states) to control virtual dials and on‑screen UI elements.[page:1]

---

## 📁 Repository Layout

```text
📦 COMPUTER_VISION-
┣ 📜 1.testt.2.py        # Preliminary environment testing
┣ 📜 2.volume_controller.py   # Hand-gesture to Pycaw volume mapping
┣ 📜 3.dialtest.3.py     # Virtual rotational dial logic
┣ 📜 4.test.4.py         # Core tracking test grounds
┣ 📜 5.cube.py           # 3D wireframe generation
┣ 📜 6.cube_proto.py     # Two-handed scaling & rotational logic
┣ 📜 7.block.click.py    # Pinch-to-click interactions
┣ 📜 8.eyedetection.py   # Base face-mesh and eye-tracking
┣ 📜 9.blink_count.py    # EAR calculation and blink iteration
┣ 📜 10.blinkcount_mod.py # Refined thresholding for blink accuracy
┣ 📜 11.dial.blink.py    # Hybrid UI: Dial control via face/hand sync
┣ 📜 constant_code.py    # Shared utility functions and constants
┣ 📜 file.py             # File handling / I/O logic
┣ 📜 face_landmarker.task # MediaPipe pre-trained face model
┣ 📜 hand_landmarker.task # MediaPipe pre-trained hand model
┗ 📜 .gitignore
