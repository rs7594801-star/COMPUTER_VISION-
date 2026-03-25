import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- Setup Mediapipe ---
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Cube Vertices ---
points = np.array([
    [-40, -40,  40], [40, -40,  40], [40,  40,  40], [-40,  40,  40],
    [-40, -40, -40], [40, -40, -40], [40,  40, -40], [-40,  40, -40]
])
edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, num_hands=1)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        overlay = img.copy() # For transparency effects

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            # Get Index finger tip for the "Anchor"
            idx = landmarks[8]
            cx, cy = int(idx.x * w), int(idx.y * h)
            
            # Calculate rotation based on Thumb (4) and Index (8)
            p4, p8 = landmarks[4], landmarks[8]
            angle_z = math.atan2((p8.y - p4.y), (p8.x - p4.x))
            angle_x = (p8.y - 0.5) * np.pi # Tilts cube as you move hand up/down

            # --- Rotation Matrices ---
            # Z-axis (Spin)
            rot_z = np.array([
                [math.cos(angle_z), -math.sin(angle_z), 0],
                [math.sin(angle_z),  math.cos(angle_z), 0],
                [0, 0, 1]
            ])
            # X-axis (Tilt)
            rot_x = np.array([
                [1, 0, 0],
                [0, math.cos(angle_x), -math.sin(angle_x)],
                [0, math.sin(angle_x),  math.cos(angle_x)]
            ])

            # Draw "Hologram Base" Ring
            cv2.circle(overlay, (cx, cy), 80, (255, 255, 0), 1)
            cv2.circle(overlay, (cx, cy), 85, (255, 255, 0), 2)

            # Project and Rotate Cube
            projected_pts = []
            for p in points:
                rotated = np.dot(rot_x, np.dot(rot_z, p))
                px = int(rotated[0] + cx)
                py = int(rotated[1] + cy)
                projected_pts.append((px, py))

            # Draw Glowing Edges
            for edge in edges:
                cv2.line(img, projected_pts[edge[0]], projected_pts[edge[1]], (255, 255, 0), 2)
            
            # Add UI Text near hand
            cv2.putText(img, f"RAD: {angle_z:.2f}", (cx+90, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(img, "SYSTEM: ACTIVE", (cx+90, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Blend overlay for that "hologram" transparency look
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow("JARVIS HUD", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()