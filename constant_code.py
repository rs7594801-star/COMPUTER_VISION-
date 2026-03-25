import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math


model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#LANDMARK START KRNE K LIYE MANUAL CODE SETUP
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    # running_mode=VisionRunningMode.VIDEO,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4)
frame = 0 

with HandLandmarker.create_from_options(options) as landmarker:
    capture = cv2.VideoCapture(0)
    if frame%2 == 0 :


        while capture is not None:
            blank , image = capture.read()
            if not blank :
                break 
            image = cv2.flip(image , 1 )
            rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            rgb = mp.Image(image_format=mp.ImageFormat.SRGB ,data = rgb )
            frame = frame_timestamp_ms = int(time.time()*1000)
            result = landmarker.detect_for_video(rgb , frame_timestamp_ms)
            # handmarkers bnane hai 
            if result.hand_landmarks :

                for hand_landmarks in  result.hand_landmarks : 
                    h , w , _ = image.shape
                    f1 =  hand_landmarks[8]
                    f2 = hand_landmarks [12]
                    f3 = hand_landmarks [16]
                    f4 =  hand_landmarks[20]
                    f5 =  hand_landmarks[4]
                    f0 =  hand_landmarks[0]
                    x1 ,y1 = int(f1.x*w) ,int (f1.y*h)
                    x2 ,y2 = int(f2.x*w) ,int (f2.y*h)
                    x3 ,y3 = int(f3.x*w) ,int (f3.y*h)
                    x4 ,y4 = int(f4.x*w) ,int (f4.y*h)
                    x5 ,y5 = int(f5.x*w) ,int (f5.y*h)
                    x0 ,y0 = int(f0.x*w) ,int (f0.y*h)

                    cv2.line(image , (x0 , y0 ),(x1 , y1 ), (0,255,0),3)
                    cv2.line(image , (x0 , y0 ),(x2 , y2 ), (0,255,0),3)
                    cv2.line(image , (x0 , y0 ),(x3 , y3 ), (0,255,0),3)
                    cv2.line(image , (x0 , y0 ),(x4 , y4 ), (0,255,0),3)
                    cv2.line(image , (x0 , y0 ),(x5 , y5 ), (0,255,0),3)
                    # length = math.hypot(x0 - x5, y0 - y5)
                    # length = math.hypot(x0 - x1, y0 - y1)
                    # length = math.hypot(x0 - x2, y0 - y2)
                    # length = math.hypot(x0 - x3, y0 - y3)
                    # length = math.hypot(x0 - x4, y0 - y4)


                    

                    for id , lm in enumerate(hand_landmarks):
                        x , y = int(lm.x * w ) , int(lm.y*h)
                        # draw circle 
                        cv2.circle(image , (x , y) , 4 , (0,255,255) , 1 )
                        
                 
                    length1 = math.hypot(x0 - x5, y0 - y5)
                    length2 = math.hypot(x0 - x1, y0 - y1)
                    length3 = math.hypot(x0 - x2, y0 - y2)
                    length4 = math.hypot(x0 - x3, y0 - y3)
                    length5 = math.hypot(x0 - x4, y0 - y4)


                    # Prepare the text (rounding the distance to 1 decimal place)
# We use f-strings to combine the label and the value
                # Color code for Red in BGR: (0, 0, 255)
                    # cv2.putText(image, f"L1: {round(length1, 1)}", (x5, y5 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(image, f"L2: {round(length2, 1)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(image, f"L3: {round(length3, 1)}", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(image, f"L4: {round(length4, 1)}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(image, f"L5: {round(length5, 1)}", (x4, y4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    if length1 > 160:
                            cv2.putText(image, f"THUMB: ", (x5, y5 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif length2 > 200:
                            cv2.putText(image, f"INDEX : ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif length3 > 150 :
                            cv2.putText(image, f"FUCK YOU : ", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif length4 > 150 :
                            cv2.putText(image, f"RING: ", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif length5 > 140 :
                             cv2.putText(image, f"PINKY : ", (x4, y4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                         
                         




                       








                    

                    
                   
            cv2.imshow("image" , image)

            # cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

capture.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import time
# import math
# import pyautogui
# import numpy as np

# # Initialize PyAutoGUI
# pyautogui.FAILSAFE = False
# screen_w, screen_h = pyautogui.size()

# model_path = 'hand_landmarker.task'
# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.VIDEO,
#     num_hands=1) # Reduced to 1 for better performance as a mouse

# frame_count = 0 

# with HandLandmarker.create_from_options(options) as landmarker:
#     capture = cv2.VideoCapture(0)
    
#     while capture.isOpened():
#         success, image = capture.read()
#         if not success: break 
        
#         image = cv2.flip(image, 1)
#         h, w, _ = image.shape
#         rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
#         timestamp = int(time.time() * 1000)
#         result = landmarker.detect_for_video(mp_image, timestamp)

#         if result.hand_landmarks:
#             for hand_landmarks in result.hand_landmarks:
#                 # 1. Get Coordinates
#                 f1 = hand_landmarks[8]  # Index
#                 f5 = hand_landmarks[4]  # Thumb
#                 f2 = hand_landmarks [12]
                
                
#                 # Convert normalized to pixel coordinates for distance calc
#                 x1, y1 = int(f1.x * w), int(f1.y * h)
#                 x5, y5 = int(f5.x * w), int(f5.y * h)
#                 x2 ,y2 = int(f2.x*w) ,int (f2.y*h)


#                 # 2. Map Coordinates to Screen
#                 # We use interp to map 640 -> 1920 and 480 -> 1080
#                 # Added a margin (70px) so you don't have to reach the very edge of the camera
#                 mouse_x = np.interp(f1.x * w, [70, w - 70], [0, screen_w])
#                 mouse_y = np.interp(f1.y * h, [70, h - 70], [0, screen_h])

#                 # Move Mouse (The index finger acts as the pointer)
#                 pyautogui.moveTo(mouse_x, mouse_y, _pause=False)

#                 # 3. Distance Logic for Click
#                 dist = math.hypot(x2 - x5, y2 - y5)
                
#                 # Visual feedback for the click zone
#                 cv2.line(image, (x1, y1), (x5, y5), (255, 0, 255), 2)
#                 cv2.line(image, (x2, y2), (x5, y5), (255, 0, 255), 2)
                
#                 if dist < 50:
#                     cv2.circle(image, (x2, y2), 15, (0,0  , 255), cv2.FILLED)
#                     pyautogui.click()
#                     # Small sleep to prevent accidental double/triple clicks
#                     time.sleep(0.1)

#                 # Draw landmarks for visual reference
#                 for lm in hand_landmarks:
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

#         cv2.imshow("Hand Mouse Control", image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# capture.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import time
# import math

# # --- Path Setup ---
# hand_model_path = 'hand_landmarker.task'
# face_model_path = 'face_landmarker.task'

# # --- MediaPipe Tasks Configuration ---
# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
# # r = mp.tasks.vision.FaceLandmarker()
# # Hand Options
# hand_options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=hand_model_path),
#     running_mode=VisionRunningMode.VIDEO,
#     num_hands=2)

# # Face Options
# face_options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=face_model_path),
#     running_mode=VisionRunningMode.VIDEO)

# # --- Start Both Landmarkers ---
# with HandLandmarker.create_from_options(hand_options) as hand_marker, \
#      FaceLandmarker.create_from_options(face_options) as face_marker:
    
#     capture = cv2.VideoCapture(0)

#     while capture.isOpened():
#         ret, image = capture.read()
#         if not ret: break

#         image = cv2.flip(image, 1)
#         h, w, _ = image.shape
        
#         # Prepare input image
#         rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#         timestamp = int(time.time() * 1000)

#         # 1. RUN DETECTION
#         hand_result = hand_marker.detect_for_video(mp_image, timestamp)
#         face_result = face_marker.detect_for_video(mp_image, timestamp)

#         # 2. MANUAL FACE MESH DRAWING (No landmark_pb2 needed)
#         if face_result.face_landmarks:
#             for face_landmarks in face_result.face_landmarks:
#                 for lm in face_landmarks:
#                     # Convert normalized coordinates to pixel coordinates
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     # Draw small dots for the mesh
#                     cv2.circle(image, (cx, cy), 1, (0, 255, 255), -1)

#         # 3. DRAW HANDS (Your Original Logic)
#         if hand_result.hand_landmarks:
#             for hand_landmarks in hand_result.hand_landmarks:
#                 # Landmark mapping from your code
#                 f1, f2, f3, f4, f5, f0 = hand_landmarks[8], hand_landmarks[12], hand_landmarks[16], hand_landmarks[20], hand_landmarks[4], hand_landmarks[0]
                
#                 x1, y1 = int(f1.x*w), int(f1.y*h)
#                 x2, y2 = int(f2.x*w), int(f2.y*h)
#                 x3, y3 = int(f3.x*w), int(f3.y*h)
#                 x4, y4 = int(f4.x*w), int(f4.y*h)
#                 x5, y5 = int(f5.x*w), int(f5.y*h)
#                 x0, y0 = int(f0.x*w), int(f0.y*h)

#                 # Lines to wrist
#                 for tx, ty in [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5)]:
#                     cv2.line(image, (x0, y0), (tx, ty), (0, 255, 0), 2)

#                 # Drawing hand dots
#                 for lm in hand_landmarks:
#                     cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 3, (255, 0, 0), -1)

#                 # Distance calculation
#                 length1 = math.hypot(x0 - x5, y0 - y5)
#                 if length1 > 160:
#                     cv2.putText(image, "THUMB", (x5, y5 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         cv2.imshow('Face Mesh + Hand Tracking', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# capture.release()
# cv2.destroyAllWindows()