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

# --- Cube Vertices (initial size) ---
initial_points = np.array([
    [-40, -40,  40], [40, -40,  40], [40,  40,  40], [-40,  40,  40],
    [-40, -40, -40], [40, -40, -40], [40,  40, -40], [-40,  40, -40]
])
edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=2 # <--- Detect two hands
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    # Initialize variables for smooth scaling
    prev_scale = 1.0
    
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        overlay = img.copy() 

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        anchor_cx, anchor_cy = w // 2, h // 2 # Default center if no hand
        current_scale = 1.0 # Default scale
        angle_z = 0
        angle_x = 0

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            # Assume first detected hand is the primary (position/rotation)
            hand1_landmarks = results.hand_landmarks[0]
            
            # Anchor point for the cube (Index finger tip of hand 1)
            idx1 = hand1_landmarks[8]
            anchor_cx, anchor_cy = int(idx1.x * w), int(idx1.y * h)
            
            # Rotation from Hand 1 (Thumb and Index)
            p4_1, p8_1 = hand1_landmarks[4], hand1_landmarks[8]
            angle_z = math.atan2((p8_1.y - p4_1.y), (p8_1.x - p4_1.x))
            angle_x = (idx1.y - 0.5) * np.pi # Tilts cube based on hand 1's vertical position

            cv2.circle(overlay, (anchor_cx, anchor_cy), 80, (255, 255, 0), 1)
            cv2.circle(overlay, (anchor_cx, anchor_cy), 85, (255, 255, 0), 2)
            cv2.putText(img, "SYSTEM: ACTIVE", (anchor_cx+90, anchor_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # --- Two-Handed Scaling ---
            if len(results.hand_landmarks) == 2:
                hand2_landmarks = results.hand_landmarks[1]
                # Use index finger tips for distance
                idx2 = hand2_landmarks[8]
                cx2, cy2 = int(idx2.x * w), int(idx2.y * h)

                # Draw connection between index fingers
                cv2.line(img, (anchor_cx, anchor_cy), (cx2, cy2), (0, 255, 255), 2)
                
                # Calculate distance between index fingertips
                distance = math.hypot(anchor_cx - cx2, anchor_cy - cy2)
                
                # Map distance to a scale factor (adjust min/max distance as needed)
                # Ensure a minimum reasonable scale
                current_scale = np.interp(distance, [50, 400], [0.5, 2.5]) # [min_dist, max_dist] -> [min_scale, max_scale]
                current_scale = max(0.2, min(current_scale, 3.0)) # Clamp scale to prevent extreme values
                prev_scale = current_scale # Update for next frame
                
                cv2.putText(img, f"SCALE: {current_scale:.2f}", (anchor_cx+90, anchor_cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                # If only one hand, maintain previous scale
                current_scale = prev_scale
                
        # --- Apply Scaling and Rotation to Cube ---
        scaled_points = initial_points * current_scale

        rot_z = np.array([
            [math.cos(angle_z), -math.sin(angle_z), 0],
            [math.sin(angle_z),  math.cos(angle_z), 0],
            [0, 0, 1]
        ])
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x),  math.cos(angle_x)]
        ])

        projected_pts = []
        for p in scaled_points:
            rotated = np.dot(rot_x, np.dot(rot_z, p))
            px = int(rotated[0] + anchor_cx)
            py = int(rotated[1] + anchor_cy)
            projected_pts.append((px, py))

        # Draw Glowing Edges
        for edge in edges:
            cv2.line(img, projected_pts[edge[0]], projected_pts[edge[1]], (255, 255, 0), 2)
            
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow("JARVIS HUD", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()