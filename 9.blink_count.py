import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

# --- 1. SETUP AND SETTINGS ---
MODEL_PATH = 'face_landmarker.task'
EAR_THRESHOLD = 0.22 
CONSEC_FRAMES = 3 

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

blink_count = 0
frame_counter = 0

# --- HELPER FUNCTION ---
def draw_eye_rect(frame, pts, eye_indices, color, w, h):
    coords = []
    for idx in eye_indices:
        point = pts[idx]
        coords.append((int(point.x * w), int(point.y * h)))
    
    coords = np.array(coords)
    x_min, y_min = coords.min(axis=0) 
    x_max, y_max = coords.max(axis=0) 
    
    padding = 5
    # color is light green: (144, 238, 144), -1 fills the box
    cv2.rectangle(frame, 
                  (x_min - padding, y_min - padding), 
                  (x_max + padding, y_max + padding), 
                  color, -1)

# --- 2. INITIALIZE THE AI ---
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
        
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp)
        detector = vision.FaceLandmarker.create_from_options(options)
        detection_result = detector.detect(frame)
        
        if result.face_landmarks:
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
            pts = result.face_landmarks[0]
            p_thumb = face_landmarks[10]
            p_index = face_landmarks[152]
                
            cx_t, cy_t = int(p_thumb.x * w), int(p_thumb.y * h)
            cx_i, cy_i = int(p_index.x * w), int(p_index.y * h)
            
            # --- CALCULATE EAR ---
            l = [np.array([pts[i].x, pts[i].y]) for i in LEFT_EYE_IDX]
            l_v1 = np.linalg.norm(l[1] - l[5])
            l_v2 = np.linalg.norm(l[2] - l[4])
            l_h  = np.linalg.norm(l[0] - l[3])
            left_ear = (l_v1 + l_v2) / (2.0 * l_h)

            r = [np.array([pts[i].x, pts[i].y]) for i in RIGHT_EYE_IDX]
            r_v1 = np.linalg.norm(r[1] - r[5])
            r_v2 = np.linalg.norm(r[2] - r[4])
            r_h  = np.linalg.norm(r[0] - r[3])
            right_ear = (r_v1 + r_v2) / (2.0 * r_h)

            avg_ear = (left_ear + right_ear) / 2.0
            
            # --- DRAW FILLED BOXES ---
            if avg_ear < EAR_THRESHOLD:
                red  = (0 , 0 , 255)
                draw_eye_rect(frame, pts, LEFT_EYE_IDX, red, w, h)
                draw_eye_rect(frame, pts, RIGHT_EYE_IDX, red , w, h)
            else:


                light_green = (144, 238, 144)
                draw_eye_rect(frame, pts, LEFT_EYE_IDX, light_green, w, h)
                draw_eye_rect(frame, pts, RIGHT_EYE_IDX, light_green, w, h)
            #      

            # --- BLINK LOGIC ---
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0
            # continue

                if avg_ear < EAR_THRESHOLD:
                    red  = (0 , 0 , 255)
                    draw_eye_rect(frame, pts, LEFT_EYE_IDX, red, w, h)
                    draw_eye_rect(frame, pts, RIGHT_EYE_IDX, red , w, h)
                else:


                    light_green = (144, 238, 144)
                    draw_eye_rect(frame, pts, LEFT_EYE_IDX, light_green, w, h)
                    draw_eye_rect(frame, pts, RIGHT_EYE_IDX, light_green, w, h)
                    # continue 
            
            # Display Count
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Blink Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()