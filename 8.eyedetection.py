


# import cv2
# import mediapipe as mp
# import numpy as np
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # 1. Configuration
# MODEL_PATH = 'face_landmarker.task'
# EAR_THRESHOLD = 0.22  # Adjust based on your eye shape
# CONSEC_FRAMES = 3     # Frames eye must be closed to count as 1 blink

# # Landmark Indices (New API indices for eyes)
# LEFT_EYE = [362, 385, 387, 263, 373, 380] # [p1, p2, p3, p4, p5, p6]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# def calculate_ear(landmarks, eye_indices):
#     # Extract coordinates
#     coords = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
#     # Vertical distances
#     v1 = np.linalg.norm(coords[1] - coords[5])
#     v2 = np.linalg.norm(coords[2] - coords[4])
#     # Horizontal distance
#     h = np.linalg.norm(coords[0] - coords[3])
#     return (v1 + v2) / (2.0 * h)

# # 2. Initialize Landmarker
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO, # Crucial for stream processing
#     num_faces=1
# )

# blink_count = 0
# frame_counter = 0

# cap = cv2.VideoCapture(0)
# with vision.FaceLandmarker.create_from_options(options) as landmarker:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         # Convert BGR to RGB and MediaPipe Image object
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
#         # Get timestamp in ms
#         timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
#         # Detect
#         result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
#         if result.face_landmarks:
#             landmarks = result.face_landmarks[0]
#             left_ear = calculate_ear(landmarks, LEFT_EYE)
#             right_ear = calculate_ear(landmarks, RIGHT_EYE)
#             avg_ear = (left_ear + right_ear) / 2.0
            
#             # Blink Detection Logic
#             if avg_ear < EAR_THRESHOLD:
#                 frame_counter += 1
#             else:
#                 if frame_counter >= CONSEC_FRAMES:
#                     blink_count += 1
#                 frame_counter = 0
            
#             cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow('Blink Counter', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup
MODEL_PATH = 'face_landmarker.task'

# These are the specific ID numbers for the dots around the eyes
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Combine them into one list to make drawing easier
ALL_EYE_POINTS = LEFT_EYE_IDX + RIGHT_EYE_IDX

# 2. Configure AI
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame  = cv2.flip(frame , 1 )
        
        # Get frame dimensions (width and height) to calculate pixel positions
        h, w, _ = frame.shape
        
        # Convert frame for the AI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Detect face
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if result.face_landmarks:
            # Get all 478 dots
            pts = result.face_landmarks[0]
            
            # --- DRAWING SECTION ---
            # Loop through only our eye index numbers
            for idx in ALL_EYE_POINTS:
                # Get the dot's coordinate (AI gives values from 0 to 1)
                point = pts[idx]
                
                # Turn the 0-1 values into actual pixel coordinates on your screen
                x_pixel = int(point.x * w)
                y_pixel = int(point.y * h)
                
                # Draw a small green circle at that pixel (radius 2, color green, filled)
                # cv2.circle(frame, (x_pixel, y_pixel), 2, (0, 255, 0), 1)
                # cv2.rectangle(frame , ())
                if result.face_landmarks:
                    pts = result.face_landmarks[0]
            
            # Helper function to get a rectangle for a set of indices
            def draw_eye_rect(eye_indices, color):
                # 1. Get all pixel coordinates for these landmarks
                coords = []
                for idx in eye_indices:
                    point = pts[idx]
                    coords.append((int(point.x * w), int(point.y * h)))
                
                # 2. Find the bounding box (min/max x and y)
                coords = np.array(coords)
                x_min , y_min = coords.min(axis=0) 
                x_max, y_max = coords.max(axis=0) 
                
                # 3. Add a little "padding" so the box isn't too tight
                padding = 5
                cv2.rectangle(frame, 
                              (x_min - padding, y_min - padding), 
                              (x_max + padding, y_max + padding), 
                              color, -1 )

            # Draw Green rectangle for Left Eye
            draw_eye_rect(LEFT_EYE_IDX, (144 ,238 ,144 ) )
            
            # Draw Blue rectangle for Right Eye
            draw_eye_rect(RIGHT_EYE_IDX, (144 , 238 , 144 ) )

            # (Optional) Keep your existing dots
            # for idx in ALL_EYE_POINTS:
            #     p = pts[idx]
            #     cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)

        # Show the result
        cv2.imshow('Only Eye Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()