import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Setup Mediapipe Tasks
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Convert and detect
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        frame_timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Get coordinates for Thumb (4) and Index (8)
                p_thumb = hand_landmarks[4]
                p_index = hand_landmarks[8]
                
                cx_t, cy_t = int(p_thumb.x * w), int(p_thumb.y * h)
                cx_i, cy_i = int(p_index.x * w), int(p_index.y * h)

                # --- ROTATION LOGIC ---
                # Calculate angle in degrees using atan2(delta_y, delta_x)
                angle_rad = math.atan2(cy_i - cy_t, cx_i - cx_t)
                angle_deg = math.degrees(angle_rad)

                # Draw the line between fingers
                cv2.line(img, (cx_t, cy_t), (cx_i, cy_i), (255, 0, 255), 3)
                cv2.circle(img, (cx_t, cy_t), 8, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (cx_i, cy_i), 8, (0, 255, 0), cv2.FILLED)

                # Visual Dial Representation
                dial_center = (w - 100, 100)
                dial_radius = 50
                cv2.circle(img, dial_center, dial_radius, (200, 200, 200), 2)
                
                # Calculate needle end point based on hand angle
                needle_x = int(dial_center[0] + dial_radius * math.cos(angle_rad))
                needle_y = int(dial_center[1] + dial_radius * math.sin(angle_rad))
                cv2.line(img, dial_center, (needle_x, needle_y), (0, 255, 255), 4)

                cv2.putText(img, f"Rotation: {int(angle_deg)} deg", (dial_center[0]-80, dial_center[1]+80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Hand Rotation Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()