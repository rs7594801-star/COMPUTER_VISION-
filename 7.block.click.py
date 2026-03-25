import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
model_path = 'hand_landmarker.task' # Path to your downloaded model
rect_pos = [100, 100]  # Initial x, y of the rectangle
rect_size = 150        # Square size
color = (255, 0, 255)  # Purple

# --- INITIALIZE MEDIAPIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) # Flip for mirror effect
    h, w, _ = frame.shape
    
    # Convert OpenCV image to MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Run Detection
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        # Get coordinates of Index Tip (8) and Middle Tip (12)
        index_tip = detection_result.hand_landmarks[0][8]
        middle_tip = detection_result.hand_landmarks[0][12]
        
        # Convert normalized coordinates to pixel values
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
        
        # 1. Check if index finger is inside the rectangle [00:09:12]
        if rect_pos[0] < ix < rect_pos[0] + rect_size and \
           rect_pos[1] < iy < rect_pos[1] + rect_size:
            
            color = (0, 255, 0) # Feedback: Turn Green [00:11:10]
            
            # 2. Check for "Click" (Distance between index and middle) [00:18:28]
            distance = ((ix - mx)**2 + (iy - my)**2)**0.5
            if distance < 25: # If fingers are close, "Drag"
                rect_pos = [ix - rect_size//2, iy - rect_size//2]
        else:
            color = (255, 0, 255) # Reset to Purple

    # Draw the rectangle
    cv2.rectangle(frame, (rect_pos[0], rect_pos[1]), 
                  (rect_pos[0] + rect_size, rect_pos[1] + rect_size), color, cv2.FILLED)

    cv2.imshow("Modern MediaPipe Drag & Drop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import time
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # --- CONFIG & STATE ---
# model_path = 'hand_landmarker.task'
# rect_pos = [100, 100]
# rect_size = 150
# color = (255, 0, 255)
# latest_result = None

# # Callback function to handle async results (Key for smoothness!)
# def update_result(result: vision.HandLandmarker , output_image: mp.Image, timestamp_ms: int):
#     global latest_result
#     latest_result = result

# # --- INITIALIZE ASYNC DETECTOR ---
# base_options = python.BaseOptions(model_asset_path=model_path)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.LIVE_STREAM, # Optimized for webcam
#     result_callback=update_result,
#     num_hands=1
# )
# detector = vision.HandLandmarker.create_from_options(options)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 60) # Set high FPS if your camera supports it

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
    
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
#     # Send frame to detector without blocking the loop
#     timestamp = int(time.time() * 1000)
#     detector.detect_async(mp_image, timestamp)
    
#     if latest_result and latest_result.hand_landmarks:
#         # Get coordinates
#         idx = latest_result.hand_landmarks[0][8]
#         mid = latest_result.hand_landmarks[0][12]
        
#         ix, iy = int(idx.x * w), int(idx.y * h)
#         mx, my = int(mid.x * w), int(mid.y * h)
        
#         # Smooth Logic: Check boundaries
#         if rect_pos[0] < ix < rect_pos[0] + rect_size and \
#            rect_pos[1] < iy < rect_pos[1] + rect_size:
            
#             color = (0, 255, 0)
            
#             # Distance for "Grab"
#             dist = ((ix - mx)**2 + (iy - my)**2)**0.5
#             if dist < 20: 
#                 # Centering: Moves the rect smoothly with the finger
#                 rect_pos[0] = ix - rect_size // 2
#                 rect_pos[1] = iy - rect_size // 2
#         else:
#             color = (255, 0, 255)

#     # Draw using OpenCV
#     cv2.rectangle(frame, (rect_pos[0], rect_pos[1]), 
#                   (rect_pos[0] + rect_size, rect_pos[1] + rect_size), color, cv2.FILLED)

#     cv2.imshow("Smooth MediaPipe Tasks", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# detector.close()
# cap.release()
# cv2.destroyAllWindows()


