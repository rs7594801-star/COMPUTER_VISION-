import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pycaw

# 1. Setup Mediapipe Tasks
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# vision.FaceLandmarkerOptions

# Initialize the landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4)
# Initialize Windows audio control
from pycaw.pycaw import AudioUtilities
device = AudioUtilities.GetSpeakers()  # Get default speakers
volume = device.EndpointVolume        # Get volume control interface
# This enables programmatic volume adjustment

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    # pTime = 0
    # mp_hands = mp.Hands


    while cap is not  None:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img,1)

        # Convert BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        imgRGB = mp.Image(image_format=mp.ImageFormat.SRGB ,  data = imgRGB)
        
        # Get timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        # frame_timestamp_ms = time()

        # Perform Detection
        results = landmarker.detect_for_video(imgRGB, frame_timestamp_ms)
        # results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # from pycaw.pycaw import AudioUtilities
        # device = AudioUtilities.GetSpeakers()
        # volume = device.EndpointVolume
        # options = HandLandmarkerOptions(
        # base_options=BaseOptions(model_asset_path=model_path),
        # running_mode=VisionRunningMode.VIDEO,
        # num_hands=4)

        # Draw Landmarks
        if results.hand_landmarks:
            # if frame_timestamp_ms == 0 :a

                for hand_landmarks in results.hand_landmarks:
                    index_finger = hand_landmarks [8]
                    h, w, _ = img.shape
                    p4 = hand_landmarks[8]
                    p8 = hand_landmarks[4 ]
                    cx4, cy4 = int(p4.x*w), int(p4.y * h)
                    cx8, cy8 = int(p8.x * w), int(p8.y * h)
                    cv2.line(img, (cx4, cy4), (cx8, cy8), (0, 255, 0), 3)
                    for id, lm in enumerate(hand_landmarks):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # Draw a circle on each joint
                        cv2.circle(img, (cx, cy), 4 , (0,255, 255), 1)
                        # 1. Calculate the distance between thumb and index finger
                    import math
                    length = math.hypot(cx8 - cx4, cy8 - cy4)

                    # 2. Get Volume Range from Pycaw
                    volRange = volume.GetVolumeRange()
                    minVol = volRange[0]
                    maxVol = volRange[1]
                    
                    # 3. Map the finger distance to the volume range
                    # Adjust '50, 300' based on how far your hand is from the camera
                    import numpy as np
                    vol = np.interp(length, [10, 200], [minVol, maxVol])
                    
                    # 4. Set the Master Volume
                    volume.SetMasterVolumeLevel(vol, None)
                    

                    # Visual feedback: Change color if "clicked" (fingers close together)
                    if length < 90:
                        cv2.circle(img,(cx4, cy4), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(img,(cx8, cy8), 10, (0, 0, 255), cv2.FILLED)
                        cv2.line(img, (cx4, cy4), (cx8, cy8), (0, 0, 255), 3)


                    
                    # Optional: You can still use the old drawing_utils if you import it
                    # or manually draw connections.
                    import numpy as np

                    # 1. Map the finger distance (length) to the bar height (pixels)
                    # Assuming your image height is 480, we use 400 as the bottom and 150 as the top
                    volBar = np.interp(length, [50, 300], [400, 150])

                    # 2. Map the finger distance to a percentage (0-100) for the text
                    volPer = np.interp(length, [50, 300], [0, 100])

                        # 3. Draw the background bar (Outline)
                    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)

                        # 4. Draw the filled volume level
                        # The color changes to green if it's "full"
                        
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

                        # 5. Add the percentage text below the bar
                    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 0, 0), 3)
                    if  int(volPer) < 50 :
                    # if int(volBar) or int(volPer) < 100 :
                        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
                        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "Rachit Bhai Volume badhade !" , (150, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                1.2, (0, 0, 255), 2)
                    else  :
                            cv2.putText(img, "AB BADHIYA HAI !" , (200, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2 , (0,255,0) , 2)
                             
                              



        # Calculate FPS
        # cTime = time.time()
        # fps = _
        # pTime = cTime

        # cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 1)
        cv2.flip(img , 1)
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()