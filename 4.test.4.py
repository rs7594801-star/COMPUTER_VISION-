import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- Mediapipe Setup ---
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- 3D Cube Vertices (x, y, z) ---
# Centered at (0,0,0) with size 100
points = np.array([
    [-50, -50,  50], [50, -50,  50], [50,  50,  50], [-50,  50,  50],
    [-50, -50, -50], [50, -50, -50], [50,  50, -50], [-50,  50, -50]
])

# Connections between vertices to draw lines (edges)
edges = [
    (0,1), (1,2), (2,3), (3,0), # Front face
    (4,5), (5,6), (6,7), (7,4), # Back face
    (0,4), (1,5), (2,6), (3,7)  # Connecting lines
]

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
        center_x, center_y = w // 2, h // 2

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        angle = 0
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            # Use Thumb (4) and Index (8) to get rotation
            p4, p8 = landmarks[4], landmarks[8]
            angle = math.atan2((p8.y - p4.y), (p8.x - p4.x))
            
            # Visual line for the "controller"
            cv2.line(img, (int(p4.x*w), int(p4.y*h)), (int(p8.x*w), int(p8.y*h)), (0, 255, 0), 2)

        # --- Rotate and Project Cube ---
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0, 0, 1]
        ])

        projected_points = []
        for p in points:
            # Rotate point
            rotated_p = np.dot(rotation_matrix, p)
            # Project to 2D screen (add center offsets)
            x = int(rotated_p[0] + center_x)
            y = int(rotated_p[1] + center_y)
            projected_points.append((x, y))

        # Draw edges
        for edge in edges:
            pt1 = projected_points[edge[0]]
            pt2 = projected_points[edge[1]]
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)

        cv2.putText(img, "Rotate Hand to Spin Cube", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("3D Cube Viewer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()